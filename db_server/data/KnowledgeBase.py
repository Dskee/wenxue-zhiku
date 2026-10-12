"""SQLAlchemy 会话与基类：知识库元数据统一存于根目录 knowledge_base.db（sqlite）。"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

# 知识库元数据数据库（与 db_server/data/info.db 不同，历史迁移后固定使用根目录 knowledge_base.db）
DB_URI = "sqlite:///./knowledge_base.db"

# check_same_thread=False：允许跨线程使用连接（FastAPI 会把同步端点丢进线程池）。
# timeout=30：SQLite 默认 5 秒就会抛 "database is locked"，并发写入时放宽等待。
engine = create_engine(
    DB_URI,
    connect_args={"check_same_thread": False, "timeout": 30},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# ⚠️ 必须是 scoped_session（按线程隔离），不能是 SessionLocal() 单例：
# 之前用模块级 session = SessionLocal() 被多个线程池工作线程并发共享，
# 而 SQLAlchemy Session 并非线程安全，会出现状态错乱 / InvalidRequestError。
# scoped_session 让每个线程自动拿到自己的 Session，调用方写法（queryall(session)）保持不变。
session = scoped_session(SessionLocal)

Base = declarative_base()
