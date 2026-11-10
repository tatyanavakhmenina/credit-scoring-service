from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CreditFeatures(BaseModel):
    """Признаки клиента: датасет UCI Default of Credit Card Clients без пола."""

    model_config = ConfigDict(extra="forbid")

    limit_bal: float = Field(gt=0, examples=[200000])
    education: int = Field(ge=0, le=6, examples=[2])
    marriage: int = Field(ge=0, le=3, examples=[1])
    age: int = Field(ge=18, le=120, examples=[35])
    pay_0: int = Field(ge=-2, le=9, examples=[0])
    pay_2: int = Field(ge=-2, le=9, examples=[0])
    pay_3: int = Field(ge=-2, le=9, examples=[0])
    pay_4: int = Field(ge=-2, le=9, examples=[0])
    pay_5: int = Field(ge=-2, le=9, examples=[0])
    pay_6: int = Field(ge=-2, le=9, examples=[0])
    bill_amt1: float = Field(examples=[50000])
    bill_amt2: float = Field(examples=[48000])
    bill_amt3: float = Field(examples=[46000])
    bill_amt4: float = Field(examples=[42000])
    bill_amt5: float = Field(examples=[40000])
    bill_amt6: float = Field(examples=[38000])
    pay_amt1: float = Field(ge=0, examples=[2000])
    pay_amt2: float = Field(ge=0, examples=[2000])
    pay_amt3: float = Field(ge=0, examples=[1500])
    pay_amt4: float = Field(ge=0, examples=[1500])
    pay_amt5: float = Field(ge=0, examples=[1500])
    pay_amt6: float = Field(ge=0, examples=[1500])


FEATURE_COLUMNS = list(CreditFeatures.model_fields)


class TaskMessage(BaseModel):
    """Команда на скоринг, которая идёт через очередь."""

    task_id: UUID
    features: CreditFeatures
    created_at: datetime


class ModelInfoResponse(BaseModel):
    name: str
    alias: str
    version: str
    run_id: str


class ProcessResponse(BaseModel):
    probability_default: float
    decision: str
    model: ModelInfoResponse


class TaskAccepted(BaseModel):
    task_id: UUID
    status: str


class TaskStatusResponse(BaseModel):
    task_id: UUID
    status: str
    probability_default: float | None
    model_version: str | None
    error: str | None
    created_at: datetime
    finished_at: datetime | None
