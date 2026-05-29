from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # Blockchain
    eth_ws_url: str = Field(
        default="wss://localhost", description="Ethereum WebSocket RPC"
    )
    eth_http_url: str = Field(
        default="https://localhost", description="Ethereum HTTP RPC"
    )
    polygon_ws_url: str = Field(
        default="wss://localhost", description="Polygon WebSocket RPC"
    )

    polygon_http_url: str = Field(
        default="https://localhost", description="Polygon HTTP RPC"
    )

    # Database
    database_url: str = Field(
        default="postgresql://user:pass@localhost:5432/cryptosentinel"
    )
    redis_url: str = Field(default="redis://localhost:6379/0")

    # Kafka
    kafka_bootstrap_servers: str = Field(default="localhost:9092")

    # Vault
    vault_addr: str = Field(default="http://localhost:8200")
    vault_token: str = Field(default="dev-token")

    # Security
    jwt_algorithm: str = Field(default="RS256")
    jwt_private_key_path: str = Field(default="keys/private.pem")
    jwt_public_key_path: str = Field(default="keys/public.pem")
    access_token_expire_minutes: int = Field(default=15)
    refresh_token_expire_days: int = Field(default=7)

    # Monitoring
    grafana_password: str = Field(default="changeme")

    # Application
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    model_artifacts_path: str = Field(default="data/models")


settings = Settings()
