"""阶段一 MVP 的应用配置。"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境变量加载的集中运行时设置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 应用配置
    app_name: str = Field(default="AI-BE-PY")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=True, alias="DEBUG")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=9020, alias="PORT")

    # ========== LLM 配置 ==========
    llm_provider: str = Field(default="tongyi", alias="LLM_PROVIDER")
    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")
    enable_mock_llm: bool = Field(default=False, alias="ENABLE_MOCK_LLM")

    # Qwen 本地模型配置
    qwen_api_key: str = Field(default="not-needed", alias="QWEN_API_KEY")
    qwen_base_url: str = Field(default="http://124.71.155.113:8080/v1", alias="QWEN_BASE_URL")
    qwen_model: str = Field(default="Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf", alias="QWEN_MODEL")

    # 通义千问 API
    tongyi_api_key: str = Field(default="sk-fe5c01bf68494deab1e458bd54895391", alias="TONGYI_API_KEY")
    tongyi_model: str = Field(default="qwen-plus", alias="TONGYI_MODEL")

    # ========== 数据库配置 ==========
    mysql_host: str = Field(default="119.3.87.115", alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")
    mysql_user: str = Field(default="root", alias="MYSQL_USER")
    mysql_password: str = Field(default="2wsx@WSX", alias="MYSQL_PASSWORD")
    mysql_database: str = Field(default="ld_test", alias="MYSQL_DATABASE")

    # ========== Redis 配置 ==========
    redis_host: str = Field(default="121.37.150.120", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=8, alias="REDIS_DB")
    redis_password: str = Field(default="3edc#EDC", alias="REDIS_PASSWORD")

    # ========== Milvus 向量数据库配置 ==========
    milvus_host: str = Field(default="60.204.236.96", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_collection: str = Field(default="ai_be_knowledge", alias="MILVUS_COLLECTION")

    # ========== OSS 对象存储配置 ==========
    oss_endpoint: str = Field(default="s3.cn-east-1.qiniucs.com", alias="OSS_ENDPOINT")
    oss_access_key: str = Field(default="VlXbgwSgTZs-aQsyzWcuo_sLD4ijmpqcbeYOPpLp", alias="OSS_ACCESS_KEY")
    oss_access_secret: str = Field(default="Fx4sYFaynPwjrezCC7UQOZmYvXZxX9EnJ8JYt4F8", alias="OSS_ACCESS_SECRET")
    oss_bucket: str = Field(default="test-public-ydd", alias="OSS_BUCKET")
    oss_domain: str = Field(default="testpublicimage.yundundun.com", alias="OSS_DOMAIN")
    local_upload_dir: str = Field(default="storage/oss", alias="LOCAL_UPLOAD_DIR")

    # ========== MCP 配置 ==========
    mcp_server_url: str = Field(default="http://localhost:8080", alias="MCP_SERVER_URL")

    # ========== 大数据 API 配置 ==========
    bigdata_api_url: str = Field(default="http://119.3.33.3:18888", alias="BIGDATA_API_URL")

    # ========== 运吨吨领舵 API 配置 ==========
    pilot_api_url: str = Field(default="http://ai-test2.yundundun.com:9002", alias="PILOT_API_URL")
    pilot_phone: str = Field(default="17327756086", alias="PILOT_PHONE")

    # ========== 地图配置 ==========
    map_base_url: str = Field(default="https://test2-map.yundundun.com", alias="MAP_BASE_URL")
    map_encrypt_key: str = Field(default="ydd2024#hips@(!.", alias="MAP_ENCRYPT_KEY")

    # ========== 多模态输入配置 ==========
    fabrx_api_key: str = Field(default="typ_live_K9itf29zqDyOl_jvH-UbCvb90c17rRPlQx75ldVI8Ek", alias="FABRX_API_KEY")
    fabrx_endpoint: str = Field(
        default="https://api.fabrx.ai/api/v1/invoice-processor-mkyw8kjg",
        alias="FABRX_ENDPOINT",
    )

    # ========== 工作记忆配置 ==========
    working_memory_enabled: bool = Field(default=True, alias="WORKING_MEMORY_ENABLED")
    working_memory_ttl_days: int = Field(default=7, alias="WORKING_MEMORY_TTL_DAYS")
    working_memory_top_n: int = Field(default=10, alias="WORKING_MEMORY_TOP_N")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回缓存的设置对象以供依赖复用。"""

    return Settings()
