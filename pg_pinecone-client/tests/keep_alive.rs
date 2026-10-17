//! A response left unread cannot return its connection to the pool. Each test counts the
//! connections a keep-alive server accepts over 20 calls of one kind: one, once every
//! response the client discards is read.

use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

use pg_pinecone_client::{
    Client, Config, DeleteDocsRequest, DeleteVectorsRequest, DocDeleteSelector,
    FetchVectorsRequest, ListVectorIdsPageRequest, QueryRequest, SearchDocsRequest,
    UpdateVectorMetadataRequest, UpsertVectorsRequest, Vector,
};
use serde_json::json;

const CALLS: usize = 20;
const NOT_FOUND: &str = r#"{"code":5,"message":"Namespace not found","details":[]}"#;

/// `(method, path) -> (status, body)` for everything but the index description.
type Route = Arc<dyn Fn(&str, &str) -> (u16, String) + Send + Sync>;

struct Server {
    url: String,
    accepted: Arc<AtomicUsize>,
}

impl Server {
    fn start(route: Route) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let url = format!("http://{}", listener.local_addr().expect("addr"));
        let accepted = Arc::new(AtomicUsize::new(0));
        let (count, host) = (accepted.clone(), url.clone());
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                count.fetch_add(1, Ordering::SeqCst);
                let (route, host) = (route.clone(), host.clone());
                thread::spawn(move || serve(stream, &route, &host));
            }
        });
        Self { url, accepted }
    }

    fn accepted(&self) -> usize {
        self.accepted.load(Ordering::SeqCst)
    }
}

/// Answers requests on one connection until the client hangs up.
fn serve(stream: TcpStream, route: &Route, host: &str) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone"));
    let mut writer = stream;
    loop {
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
            return;
        }
        let mut words = request_line.split_whitespace();
        let method = words.next().unwrap_or_default().to_string();
        let target = words.next().unwrap_or_default().to_string();
        let path = target.split('?').next().unwrap_or_default().to_string();
        let mut length = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                return;
            }
            if line == "\r\n" {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                length = value.trim().parse().unwrap_or(0);
            }
        }
        let mut body = vec![0u8; length];
        if reader.read_exact(&mut body).is_err() {
            return;
        }
        let (status, text) = if method == "GET" && path == "/indexes/idx" {
            (200, json!({"host": host}).to_string())
        } else {
            route(&method, &path)
        };
        let reply = format!(
            "HTTP/1.1 {status} Reply\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{text}",
            text.len()
        );
        if writer.write_all(reply.as_bytes()).is_err() {
            return;
        }
    }
}

fn config(server: &Server) -> Config {
    Config {
        control_plane_url: server.url.clone(),
        api_key: "key".to_string(),
        timeout: Duration::from_secs(5),
        user_agent: None,
    }
}

/// Connections accepted over [`CALLS`] runs of `call` against a data plane that answers every
/// request with `status` and `body`.
fn connections(status: u16, body: &str, call: impl Fn(&Client)) -> usize {
    let body = body.to_string();
    let server = Server::start(Arc::new(move |_, _| (status, body.clone())));
    let client = Client::new(config(&server));
    for _ in 0..CALLS {
        call(&client);
    }
    server.accepted()
}

#[test]
fn deleting_vectors_reuses_the_connection() {
    let accepted = connections(200, "{}", |client| {
        client
            .delete_vectors(&DeleteVectorsRequest {
                index: "idx".to_string(),
                namespace: Some("ns".to_string()),
                ids: Some(vec!["1".to_string()]),
                delete_all: false,
                filter: None,
            })
            .expect("delete");
    });
    assert_eq!(accepted, 1, "connections for {CALLS} deletes");
}

#[test]
fn deleting_vectors_of_a_missing_namespace_reuses_the_connection() {
    let accepted = connections(404, NOT_FOUND, |client| {
        client
            .delete_vectors(&DeleteVectorsRequest {
                index: "idx".to_string(),
                namespace: Some("ns".to_string()),
                ids: None,
                delete_all: true,
                filter: None,
            })
            .expect("a namespace that is not there is deleted");
    });
    assert_eq!(accepted, 1, "connections for {CALLS} wipes");
}

#[test]
fn deleting_a_namespace_reuses_the_connection() {
    let accepted = connections(200, "{}", |client| {
        client.delete_namespace("idx", "ns").expect("delete");
    });
    assert_eq!(accepted, 1, "connections for {CALLS} namespace deletes");
}

#[test]
fn deleting_documents_reuses_the_connection() {
    let accepted = connections(202, "{}", |client| {
        client
            .delete_docs(&DeleteDocsRequest {
                index: "idx".to_string(),
                namespace: Some("ns".to_string()),
                selector: DocDeleteSelector::Ids(vec!["1".to_string()]),
            })
            .expect("delete");
    });
    assert_eq!(accepted, 1, "connections for {CALLS} document deletes");
}

#[test]
fn a_query_of_a_missing_namespace_reuses_the_connection() {
    let accepted = connections(404, NOT_FOUND, |client| {
        let matches = client
            .query(&QueryRequest {
                index: "idx".to_string(),
                vector: vec![0.5, 0.25],
                top_k: 3,
                namespace: Some("ns".to_string()),
                filter: None,
                include_metadata: false,
                scan_factor: None,
                max_candidates: None,
            })
            .expect("no matches");
        assert!(matches.is_empty());
    });
    assert_eq!(accepted, 1, "connections for {CALLS} empty queries");
}

#[test]
fn a_listing_of_a_missing_namespace_reuses_the_connection() {
    let accepted = connections(404, NOT_FOUND, |client| {
        let page = client
            .list_vector_ids_page(&ListVectorIdsPageRequest {
                index: "idx".to_string(),
                namespace: Some("ns".to_string()),
                pagination_token: None,
            })
            .expect("an empty page");
        assert!(page.ids.is_empty());
    });
    assert_eq!(accepted, 1, "connections for {CALLS} empty listings");
}

#[test]
fn a_fetch_from_a_missing_namespace_reuses_the_connection() {
    let accepted = connections(404, NOT_FOUND, |client| {
        let found = client
            .fetch_vectors(&FetchVectorsRequest {
                index: "idx".to_string(),
                namespace: Some("ns".to_string()),
                ids: vec!["1".to_string()],
            })
            .expect("nothing found");
        assert!(found.is_empty());
    });
    assert_eq!(accepted, 1, "connections for {CALLS} empty fetches");
}

#[test]
fn a_document_search_of_a_missing_namespace_reuses_the_connection() {
    let accepted = connections(404, NOT_FOUND, |client| {
        let matches = client
            .search_docs(&SearchDocsRequest {
                index: "idx".to_string(),
                namespace: Some("ns".to_string()),
                score_by: json!([{"type": "query_string", "query": "x"}]),
                filter: None,
                include_fields: vec!["title".to_string()],
                top_k: 3,
            })
            .expect("no matches");
        assert!(matches.is_empty());
    });
    assert_eq!(accepted, 1, "connections for {CALLS} empty searches");
}

#[test]
fn a_metadata_update_reuses_the_connection() {
    let accepted = connections(200, "{}", |client| {
        client
            .update_vector_metadata(&UpdateVectorMetadataRequest {
                index: "idx".to_string(),
                namespace: Some("ns".to_string()),
                id: "1".to_string(),
                metadata: json!({"a": 1}),
            })
            .expect("update");
    });
    assert_eq!(accepted, 1, "connections for {CALLS} updates");
}

/// Control: a response the client reads already keeps the connection.
#[test]
fn an_upsert_reuses_the_connection() {
    let accepted = connections(200, r#"{"upsertedCount":1}"#, |client| {
        client
            .upsert_vectors(&UpsertVectorsRequest {
                index: "idx".to_string(),
                namespace: Some("ns".to_string()),
                vectors: vec![Vector {
                    id: "1".to_string(),
                    values: vec![0.5, 0.25],
                    metadata: None,
                }],
            })
            .expect("upsert");
    });
    assert_eq!(accepted, 1, "connections for {CALLS} upserts");
}

/// The 429 the client retries is discarded too.
#[test]
fn a_retried_upsert_reuses_the_connection() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let server = Server::start(Arc::new(move |_, _| {
        match seen.fetch_add(1, Ordering::SeqCst) % 2 {
            0 => (429, r#"{"message":"slow down"}"#.to_string()),
            _ => (200, r#"{"upsertedCount":1}"#.to_string()),
        }
    }));
    let client = Client::with_sleep(config(&server), |_| {});
    for _ in 0..CALLS {
        client
            .upsert_vectors(&UpsertVectorsRequest {
                index: "idx".to_string(),
                namespace: Some("ns".to_string()),
                vectors: vec![Vector {
                    id: "1".to_string(),
                    values: vec![0.5, 0.25],
                    metadata: None,
                }],
            })
            .expect("the retry succeeds");
    }
    assert_eq!(
        server.accepted(),
        1,
        "connections for {CALLS} retried upserts"
    );
}
