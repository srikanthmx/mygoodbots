CREATE TABLE IF NOT EXISTS conversation_turns (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    bot_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    bot_reply TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_session_id ON conversation_turns (session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_turns_bot_id ON conversation_turns (bot_id);
