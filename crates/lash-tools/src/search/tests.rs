#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn grep_provider_with_base_path(base_path: std::path::PathBuf) -> StaticToolProvider<Grep> {
        StaticToolProvider::new(
            vec![grep_tool_definition()],
            Grep::with_base_path(base_path),
        )
    }

    #[test]
    fn grep_uses_limit_argument_in_model_contract() {
        let definition = grep_tool_definition();
        let properties = definition
            .contract
            .input_schema
            .get("properties")
            .and_then(serde_json::Value::as_object)
            .expect("object properties");

        assert!(properties.contains_key("limit"));
        assert!(!properties.contains_key("maxResults"));
        assert_eq!(properties["limit"]["default"], serde_json::json!(20));
    }

    #[test]
    fn grep_contract_documents_result_shape() {
        let definition = grep_tool_definition();

        assert_eq!(definition.contract.output_schema["type"], json!("object"));
        assert!(definition.contract.output_schema["properties"]["matches"].is_object());
        assert!(definition.contract.output_schema["properties"]["count"].is_object());
        assert!(definition.contract.output_schema["properties"]["cursor"].is_object());
        let rendered = definition.compact_contract().render_signature();
        assert!(rendered.contains("matches"), "{rendered}");
        assert!(rendered.contains("count"), "{rendered}");
    }

    #[tokio::test]
    async fn test_grep_matches_with_query() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("test.txt"),
            "hello world\nfoo bar\nhello again\n",
        )
        .unwrap();

        let tool = grep_provider_with_base_path(dir.path().to_path_buf());
        let result = lash_core::testing::run_tool(&tool, "grep", &json!({"query": "hello"})).await;
        assert!(result.is_success());
        assert_eq!(result.value_for_projection()["count"], 2);
        assert_eq!(
            result.value_for_projection()["matches"][0]["path"],
            "test.txt"
        );
        assert_eq!(
            result.value_for_projection()["matches"][0]["excerpt"],
            "hello world"
        );
        assert_eq!(
            result.value_for_projection()["matches"][1]["excerpt"],
            "hello again"
        );
    }

    #[tokio::test]
    async fn test_grep_returns_structured_file_summaries() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "fn thing() {}\n").unwrap();

        let tool = grep_provider_with_base_path(dir.path().to_path_buf());
        let result = lash_core::testing::run_tool(&tool, "grep", &json!({"query": "thing"})).await;
        assert!(result.is_success());
        assert_eq!(
            result.value_for_projection()["files"][0]["path"],
            "alpha.rs"
        );
        assert_eq!(result.value_for_projection()["files"][0]["count"], 1);
        assert_eq!(result.value_for_projection()["suggested_path"], "alpha.rs");
    }

    #[tokio::test]
    async fn test_grep_structured_counts() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "ctx\nctx\n").unwrap();

        let tool = grep_provider_with_base_path(dir.path().to_path_buf());
        let result = lash_core::testing::run_tool(&tool, "grep", &json!({"query": "ctx"})).await;
        assert!(result.is_success());
        assert_eq!(result.value_for_projection()["count"], 2);
        assert_eq!(result.value_for_projection()["files"][0]["count"], 2);
    }

    #[tokio::test]
    async fn test_grep_empty_result_keeps_structured_metadata() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "ctx\n").unwrap();

        let tool = grep_provider_with_base_path(dir.path().to_path_buf());
        let result =
            lash_core::testing::run_tool(&tool, "grep", &json!({"query": "missing"})).await;
        assert!(result.is_success());
        assert_eq!(
            result.value_for_projection()["matches"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
        assert!(result.value_for_projection()["broadened_from"].is_null());
        assert!(result.value_for_projection()["regex_fallback_error"].is_null());
    }

    #[tokio::test]
    async fn test_grep_long_query_does_not_panic_in_fuzzy_fallback() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "short searchable content\n").unwrap();

        let query = "definitely missing ".repeat(20);
        let tool = grep_provider_with_base_path(dir.path().to_path_buf());
        let result = lash_core::testing::run_tool(&tool, "grep", &json!({"query": query})).await;

        assert!(
            result.is_success(),
            "long query should not panic or fail: {:?}",
            result.value_for_projection()
        );
    }

    #[test]
    fn test_cleanup_fuzzy_query_caps_to_fff_score_limit() {
        let query = "Ä".repeat(MAX_FFF_FUZZY_QUERY_BYTES + 10);
        let cleaned = cleanup_fuzzy_query(&query);

        assert!(cleaned.len() <= MAX_FFF_FUZZY_QUERY_BYTES);
        assert!(cleaned.is_char_boundary(cleaned.len()));
    }

    #[tokio::test]
    async fn test_grep_initializes_backend_lazily() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "ctx\n").unwrap();

        let tool = grep_provider_with_base_path(dir.path().to_path_buf());
        assert!(tool.executor().backend.get().is_none());

        let result = lash_core::testing::run_tool(&tool, "grep", &json!({"query": "ctx"})).await;
        assert!(result.is_success());
        assert!(tool.executor().backend.get().is_some());
    }

    #[tokio::test]
    async fn test_grep_path_scopes_search_to_subdirectory() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("inner")).unwrap();
        std::fs::write(dir.path().join("outer.txt"), "banana at root\n").unwrap();
        std::fs::write(dir.path().join("inner/inner.txt"), "banana in inner\n").unwrap();

        let tool = grep_provider_with_base_path(dir.path().to_path_buf());
        let result = lash_core::testing::run_tool(
            &tool,
            "grep",
            &json!({"query": "banana", "path": "inner"}),
        )
        .await;
        assert!(result.is_success());
        assert!(
            result.value_for_projection()["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "inner.txt"),
            "expected inner.txt match, got {:?}",
            result.value_for_projection()
        );
        assert!(
            !result.value_for_projection()["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "outer.txt"),
            "path scope should exclude outer.txt, got {:?}",
            result.value_for_projection()
        );
    }

    #[tokio::test]
    async fn test_grep_path_constrains_search_to_single_file() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("notes.txt"), "banana\n").unwrap();
        std::fs::write(dir.path().join("other.txt"), "banana\n").unwrap();

        let tool = grep_provider_with_base_path(dir.path().to_path_buf());
        let result = lash_core::testing::run_tool(
            &tool,
            "grep",
            &json!({"query": "banana", "path": "notes.txt"}),
        )
        .await;
        assert!(result.is_success());
        assert!(
            result.value_for_projection()["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "notes.txt"),
            "expected notes.txt match, got {:?}",
            result.value_for_projection()
        );
        assert!(
            !result.value_for_projection()["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "other.txt"),
            "file path should exclude other.txt"
        );
        assert!(
            tool.executor().backend.get().is_none(),
            "single-file grep should bypass the indexed backend"
        );
        assert_eq!(result.value_for_projection()["timed_out"], false);
        assert_eq!(
            result.value_for_projection()["error"],
            serde_json::Value::Null
        );
    }

    #[tokio::test]
    async fn test_grep_file_path_uses_direct_scan_for_multiword_query() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("bottle.py"),
            "header cookie static_file abort redirect request response\nunrelated\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("other.py"),
            "header cookie static_file abort redirect request response\n",
        )
        .unwrap();

        let tool = grep_provider_with_base_path(dir.path().to_path_buf());
        let result = lash_core::testing::run_tool(
            &tool,
            "grep",
            &json!({
                "query": "header cookie static_file abort redirect request response",
                "path": "bottle.py",
                "limit": 80,
            }),
        )
        .await;

        assert!(
            result.is_success(),
            "direct grep failed: {:?}",
            result.value_for_projection()
        );
        assert_eq!(result.value_for_projection()["count"], 1);
        assert_eq!(result.value_for_projection()["shown"], 1);
        assert_eq!(
            result.value_for_projection()["matches"][0]["path"],
            "bottle.py"
        );
        assert_eq!(
            result.value_for_projection()["matches"][0]["match"],
            "header cookie static_file abort redirect request response"
        );
        assert!(
            tool.executor().backend.get().is_none(),
            "single-file grep should not initialize fff"
        );
        assert_eq!(result.value_for_projection()["timed_out"], false);
        assert_eq!(
            result.value_for_projection()["error"],
            serde_json::Value::Null
        );
    }

    #[tokio::test]
    async fn test_grep_path_can_search_outside_workspace() {
        let workspace = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("external.txt"), "banana\n").unwrap();

        let tool = grep_provider_with_base_path(workspace.path().to_path_buf());
        let result = lash_core::testing::run_tool(
            &tool,
            "grep",
            &json!({
                "query": "banana",
                "path": outside.path().to_string_lossy(),
            }),
        )
        .await;
        assert!(
            result.is_success(),
            "expected search outside workspace to succeed, got {:?}",
            result.value_for_projection()
        );
        assert!(
            result.value_for_projection()["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "external.txt"),
            "expected external.txt match, got {:?}",
            result.value_for_projection()
        );
    }

    #[tokio::test]
    async fn test_grep_infers_obvious_path_prefix_from_query() {
        let workspace = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("external.txt"), "banana\n").unwrap();

        let tool = grep_provider_with_base_path(workspace.path().to_path_buf());
        let result = lash_core::testing::run_tool(
            &tool,
            "grep",
            &json!({"query": format!("{} banana", outside.path().display())}),
        )
        .await;
        assert!(result.is_success());
        assert!(
            result.value_for_projection()["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "external.txt"),
            "expected inferred path search to find external.txt, got {:?}",
            result.value_for_projection()
        );
    }

    #[tokio::test]
    async fn test_grep_infers_obvious_file_prefix_without_indexing() {
        let workspace = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let file = outside.path().join("external.txt");
        std::fs::write(&file, "banana split\n").unwrap();

        let tool = grep_provider_with_base_path(workspace.path().to_path_buf());
        let result = lash_core::testing::run_tool(
            &tool,
            "grep",
            &json!({"query": format!("{} banana", file.display())}),
        )
        .await;
        assert!(result.is_success());
        assert_eq!(
            result.value_for_projection()["matches"][0]["path"],
            "external.txt"
        );
        assert!(
            tool.executor().backend.get().is_none(),
            "inferred single-file grep should bypass fff"
        );
    }

    #[test]
    fn test_direct_file_grep_observes_pre_cancelled_abort_signal() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("notes.txt");
        std::fs::write(&file, "banana\n").unwrap();
        let abort = AtomicBool::new(true);

        let result = direct_file_grep_sync("banana", &file, Some(dir.path()), 20, &abort);

        assert!(!result.is_success());
        let value = result.value_for_projection();
        assert_eq!(value["cancelled"], true);
        assert_eq!(value["error"]["kind"], "cancelled");
        let output = result.as_output().value_for_projection();
        assert_eq!(output["message"], "grep cancelled");
        assert_eq!(output["source"], "cancellation");
    }

    #[tokio::test]
    async fn test_grep_path_missing_returns_clear_error() {
        let workspace = TempDir::new().unwrap();
        let tool = grep_provider_with_base_path(workspace.path().to_path_buf());
        let result = lash_core::testing::run_tool(
            &tool,
            "grep",
            &json!({"query": "banana", "path": "/nonexistent/totally/fake"}),
        )
        .await;
        assert!(!result.is_success());
        let value = result.value_for_projection();
        let message = value.as_str().unwrap_or("");
        assert!(
            message.contains("does not exist"),
            "expected missing-path error, got {message:?}"
        );
    }

    #[tokio::test]
    async fn test_grep_backend_is_shared_process_wide_for_same_workspace() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "ctx\n").unwrap();

        let left = Grep::with_base_path(dir.path().to_path_buf());
        let right = Grep::with_base_path(dir.path().to_path_buf());

        let left_backend = left.ensure_ready_for_query("ctx").expect("left backend");
        let right_backend = right.ensure_ready_for_query("ctx").expect("right backend");

        assert!(Arc::ptr_eq(&left_backend, &right_backend));
    }
}
