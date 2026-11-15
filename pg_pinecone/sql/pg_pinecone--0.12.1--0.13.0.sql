-- 0.12.1 -> 0.13.0: pinecone.doctor(), the one-call install and health check.
CREATE FUNCTION pinecone."doctor"(
	"remote" bool DEFAULT true /* Option < bool > */
) RETURNS TABLE (
	"check_name" TEXT,  /* String */
	"status" TEXT,  /* String */
	"detail" TEXT,  /* String */
	"hint" TEXT  /* Option < String > */
)
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'doctor_wrapper';
