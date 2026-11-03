-- Apply using psql with ON_ERROR_STOP enabled.
BEGIN;
SELECT pg_advisory_xact_lock(145991, 1);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version integer PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = 1) THEN
        CREATE TABLE conversation_sessions (
            id uuid PRIMARY KEY,
            created_at timestamptz NOT NULL DEFAULT now()
        );

        -- Each record contains a complete PydanticAI turn, including tool messages.
        -- Allocate turn numbers while holding a lock on the parent session row.
        CREATE TABLE conversation_turns (
            session_id uuid NOT NULL REFERENCES conversation_sessions(id) ON DELETE CASCADE,
            turn_number bigint NOT NULL CHECK (turn_number > 0),
            messages jsonb NOT NULL CHECK (
                jsonb_typeof(messages) = 'array' AND messages <> '[]'::jsonb
            ),
            created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (session_id, turn_number)
        );

        INSERT INTO schema_migrations (version) VALUES (1);
    END IF;
END $$;
COMMIT;
