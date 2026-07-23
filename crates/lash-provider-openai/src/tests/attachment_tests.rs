use super::*;

#[test]
fn responses_pdf_url_serializes_as_input_file_url() {
    let provider = OpenAiProvider::new("key");
    let mut req = request(vec![LlmMessage::new(
        LlmRole::User,
        vec![LlmContentBlock::Attachment { attachment_idx: 0 }],
    )]);
    req.attachments = vec![AttachmentSource::external_url(
        lash_core::MediaType::parse("application/pdf").unwrap(),
        "https://example.test/report.pdf",
    )];

    let body = provider.build_responses_request_body(&req, false).unwrap();
    assert_eq!(body["input"][0]["content"][0]["type"], "input_file");
    assert_eq!(
        body["input"][0]["content"][0]["file_url"],
        "https://example.test/report.pdf"
    );
}
