"""知识库元数据表的 CRUD：add / delete / queryall。"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime
from .KnowledgeBase import Base, engine


class Knowledge(Base):
    __tablename__ = "knowledge"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(String(500), default="")
    created_at = Column(DateTime, default=datetime.now)


Base.metadata.create_all(engine)


def add(session, name: str, description: str = "") -> str:
    exists = session.query(Knowledge).filter(Knowledge.name == name).first()
    if exists:
        return f"知识库 '{name}' 已存在"
    session.add(Knowledge(name=name, description=description))
    session.commit()
    return f"知识库 '{name}' 添加成功"


def delete(session, name: str) -> str:
    obj = session.query(Knowledge).filter(Knowledge.name == name).first()
    if not obj:
        return f"知识库 '{name}' 不存在"
    session.delete(obj)
    session.commit()
    return f"知识库 '{name}' 删除成功"


def queryall(session):
    rows = session.query(Knowledge).all()
    return [{"name": r.name, "description": r.description or ""} for r in rows]
