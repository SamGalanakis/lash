use std::fmt;

#[derive(
    Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, PartialOrd, Ord,
)]
#[serde(transparent)]
pub struct AttachmentId(String);

impl AttachmentId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AttachmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for AttachmentId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for AttachmentId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageMediaType {
    Png,
    Jpeg,
    Gif,
    Webp,
    Bmp,
}

impl ImageMediaType {
    pub fn from_mime(mime: &str) -> Option<Self> {
        match mime.trim().to_ascii_lowercase().as_str() {
            "image/png" => Some(Self::Png),
            "image/jpeg" | "image/jpg" => Some(Self::Jpeg),
            "image/gif" => Some(Self::Gif),
            "image/webp" => Some(Self::Webp),
            "image/bmp" => Some(Self::Bmp),
            _ => None,
        }
    }

    pub fn canonical_mime(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Gif => "image/gif",
            Self::Webp => "image/webp",
            Self::Bmp => "image/bmp",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "type", rename_all = "snake_case")]
pub enum MediaType {
    Image(ImageMediaType),
}

impl MediaType {
    pub fn from_mime(mime: &str) -> Option<Self> {
        ImageMediaType::from_mime(mime).map(Self::Image)
    }

    pub fn canonical_mime(self) -> &'static str {
        match self {
            Self::Image(image) => image.canonical_mime(),
        }
    }

    pub fn is_image(self) -> bool {
        matches!(self, Self::Image(_))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AttachmentCreateMeta {
    pub media_type: MediaType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl AttachmentCreateMeta {
    pub fn new(
        media_type: MediaType,
        width: Option<u32>,
        height: Option<u32>,
        label: Option<String>,
    ) -> Self {
        Self {
            media_type,
            width,
            height,
            label,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AttachmentMeta {
    pub id: AttachmentId,
    pub media_type: MediaType,
    pub byte_len: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl AttachmentMeta {
    pub fn new(
        id: AttachmentId,
        media_type: MediaType,
        byte_len: u64,
        width: Option<u32>,
        height: Option<u32>,
        label: Option<String>,
    ) -> Self {
        Self {
            id,
            media_type,
            byte_len,
            width,
            height,
            label,
        }
    }

    pub fn as_ref(&self) -> AttachmentRef {
        AttachmentRef {
            id: self.id.clone(),
            media_type: self.media_type,
            byte_len: self.byte_len,
            width: self.width,
            height: self.height,
            label: self.label.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AttachmentRef {
    pub id: AttachmentId,
    pub media_type: MediaType,
    pub byte_len: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl AttachmentRef {
    pub fn meta(&self) -> AttachmentMeta {
        AttachmentMeta {
            id: self.id.clone(),
            media_type: self.media_type,
            byte_len: self.byte_len,
            width: self.width,
            height: self.height,
            label: self.label.clone(),
        }
    }

    pub fn canonical_mime(&self) -> &'static str {
        self.media_type.canonical_mime()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_mime_parses_to_canonical_type() {
        assert_eq!(
            MediaType::from_mime("image/jpg")
                .expect("jpeg")
                .canonical_mime(),
            "image/jpeg"
        );
        assert_eq!(
            MediaType::from_mime("image/webp")
                .expect("webp")
                .canonical_mime(),
            "image/webp"
        );
        assert!(MediaType::from_mime("application/octet-stream").is_none());
    }
}
