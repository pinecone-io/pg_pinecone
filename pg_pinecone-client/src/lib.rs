//! Pure-Rust synchronous client for the Pinecone control and data planes.

mod client;
mod config;
mod error;
mod transport;
mod types;

pub use client::{
    API_VERSION, BACKUP_API_VERSION, Client, DOC_API_VERSION, VERSION, names_namespace_limit,
    normalize_scheme,
};
pub use config::Config;
pub use error::Error;
pub use types::{
    BackupInfo, CreateBackupRequest, CreateDocIndexRequest, CreateIndexRequest, DeleteDocsRequest,
    DeleteVectorsRequest, DocDeleteSelector, DocIndexInfo, DocMatch, EmbedRequest,
    FetchByMetadataPage, FetchByMetadataRequest, FetchVectorsRequest, IndexInfo, IndexStats,
    ListNamespacesPage, ListNamespacesRequest, ListVectorIdsPage, ListVectorIdsPageRequest,
    ListVectorsRequest, Match, NamespaceInfo, NamespaceStats, QueryRequest, ReadCapacity,
    RerankRequest, RerankResult, SearchDocsRequest, UPSERT_MAX_BYTES, UPSERT_MAX_RECORDS,
    UpdateVectorMetadataRequest, UpsertDocsRequest, UpsertVectorsRequest, Vector,
};
