use std::error::Error;
use std::fmt::*;

pub struct ImgError {
    repr: Repr,
}

impl Error for ImgError {}

impl Display for ImgError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Display::fmt(&self.repr, f)
    }
}

impl Debug for ImgError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Debug::fmt(&self.repr, f)
    }
}

impl ImgError {
    pub fn new<E>(kind: ImgErrorKind, error: E) -> ImgError
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::_new(kind, error.into())
    }

    fn _new(kind: ImgErrorKind, error: Box<dyn std::error::Error + Send + Sync>) -> ImgError {
        ImgError {
            repr: Repr::Custom(Box::new(Custom { kind, error })),
        }
    }

    #[inline]
    pub const fn new_const(kind: ImgErrorKind, message: String) -> ImgError {
        Self {
            repr: Repr::SimpleMessage(kind, message),
        }
    }

    #[must_use]
    #[inline]
    pub fn raw_os_error(&self) -> Option<i32> {
        match self.repr {
            Repr::Custom(..) => None,
            Repr::SimpleMessage(..) => None,
        }
    }

    #[must_use]
    #[inline]
    pub fn get_ref(&self) -> Option<&(dyn std::error::Error + Send + Sync + 'static)> {
        match self.repr {
            Repr::SimpleMessage(..) => None,
            Repr::Custom(ref c) => Some(&*c.error),
        }
    }

    pub fn into_inner(self) -> Option<Box<dyn std::error::Error + Send + Sync>> {
        match self.repr {
            Repr::SimpleMessage(..) => None,
            Repr::Custom(c) => Some(c.error),
        }
    }
}

pub(crate) enum Repr {
    SimpleMessage(ImgErrorKind, String),
    Custom(Box<Custom>),
}

impl Display for Repr {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::SimpleMessage(kind, message) => write!(f, "{}: {}", kind.as_str(), message),
            Self::Custom(custom) => write!(f, "{}: {}", custom.kind.as_str(), custom.error),
        }
    }
}

impl Debug for Repr {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::SimpleMessage(kind, message) => f
                .debug_struct("SimpleMessage")
                .field("kind", kind)
                .field("message", message)
                .finish(),
            Self::Custom(custom) => f
                .debug_struct("Custom")
                .field("kind", &custom.kind)
                .field("error", &custom.error)
                .finish(),
        }
    }
}

#[allow(unused)]
#[derive(Debug)]
pub(crate) struct Custom {
    kind: ImgErrorKind,
    error: Box<dyn std::error::Error + Send + Sync>,
}

#[allow(unused)]
#[derive(Debug)]
pub enum ImgErrorKind {
    UnknownFormat,
    OutOfMemory,
    CannotDecode,
    CannotEncode,
    MemoryOfShortage,
    SizeZero,
    NoSupportFormat,
    UnimprimentFormat,
    IllegalData,
    DecodeError,
    EncodeError,
    WriteError,
    IOError,
    OutboundIndex,
    Reset,
    IllegalCallback,
    NotInitializedImageBuffer,
    OSError,
    UnknownError,
}

#[allow(unused)]
impl ImgErrorKind {
    pub(crate) fn as_str(&self) -> &'static str {
        use ImgErrorKind::*;
        match self {
            UnknownFormat => "Unknown format",
            OutOfMemory => "Out of memory",
            CannotDecode => "Cannot decode this decoder",
            CannotEncode => "Cannot encode this encoder",
            MemoryOfShortage => "Memroy shortage",
            SizeZero => "size is zero",
            NoSupportFormat => "No Support format",
            UnimprimentFormat => "Unimplement format",
            IllegalData => "Illegal data",
            DecodeError => "decode error",
            EncodeError => "encode error",
            WriteError => "write error",
            IOError => "IO error",
            Reset => "Decoder Reset command",
            OutboundIndex => "Outbound index",
            IllegalCallback => "Illegal Callback",
            NotInitializedImageBuffer => "Not initialized Image Buffer",
            OSError => "OS error",
            UnknownError => "Unkonw error",
        }
    }
}
