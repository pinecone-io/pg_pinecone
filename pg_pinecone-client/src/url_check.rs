//! Scheme policy for every URL the API key is sent to.

use std::net::{Ipv4Addr, Ipv6Addr};

use crate::Error;

/// Accepts `https://` URLs, and `http://` only for a loopback host (`localhost`,
/// `127.0.0.0/8`, `::1`) so local mocks work. Anything else would put the API key
/// on the wire in cleartext. The error never echoes userinfo or the path.
pub fn require_secure_url(url: &str) -> Result<(), Error> {
    let bad = |reason: String| Error::InsecureUrl { reason };
    if url
        .chars()
        .any(|c| c.is_control() || c.is_whitespace() || c == '\\')
    {
        return Err(bad(
            "URL contains whitespace, control or backslash characters".into(),
        ));
    }
    let Some((scheme, rest)) = url.split_once("://") else {
        return Err(bad("URL must start with https://".into()));
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let hostport = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let host = match hostport.strip_prefix('[') {
        Some(v6) => v6.split_once(']').map_or("", |(h, _)| h),
        None => hostport.split(':').next().unwrap_or(""),
    };
    if host.is_empty() {
        return Err(bad("URL has no host".into()));
    }
    if scheme.eq_ignore_ascii_case("https") {
        return Ok(());
    }
    if scheme.eq_ignore_ascii_case("http") {
        return if is_loopback(host) {
            Ok(())
        } else {
            Err(bad(format!(
                "http:// is only allowed for loopback hosts, not '{host}'; use https://"
            )))
        };
    }
    Err(bad(format!("unsupported scheme '{scheme}'; use https://")))
}

fn is_loopback(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<Ipv4Addr>().is_ok_and(|ip| ip.is_loopback())
        || host.parse::<Ipv6Addr>().is_ok_and(|ip| ip.is_loopback())
}

#[cfg(test)]
mod tests {
    use super::require_secure_url;

    #[test]
    fn accepts() {
        for url in [
            "https://api.pinecone.io",
            "HTTPS://api.pinecone.io/",
            "https://example.com:8443/x",
            "http://localhost",
            "http://LOCALHOST:8080",
            "http://127.0.0.1:5555",
            "http://127.1.2.3",
            "http://[::1]:9000/path",
        ] {
            assert!(require_secure_url(url).is_ok(), "{url}");
        }
    }

    #[test]
    fn rejects() {
        for url in [
            "http://api.pinecone.io",
            "http://example.com",
            "http://10.0.0.1",
            "http://128.0.0.1",
            "http://0.0.0.0",
            "http://[::2]",
            "http://localhost.evil.com",
            "http://127.0.0.1.evil.com",
            "http://127.0.0.1@evil.com",
            "http://localhost:80@evil.com/",
            "http://evil.com\\@127.0.0.1",
            "http://evil.com#@localhost",
            "http://",
            "https://",
            "ftp://localhost",
            "api.pinecone.io",
            "",
            "https://exa mple.com",
        ] {
            assert!(require_secure_url(url).is_err(), "{url}");
        }
    }

    #[test]
    fn error_hides_userinfo() {
        let msg = require_secure_url("http://user:secret@example.com/x")
            .unwrap_err()
            .to_string();
        assert!(!msg.contains("secret"), "{msg}");
    }
}
