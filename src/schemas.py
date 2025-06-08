"""
OASF Schema Extensions for strongly-typed business logic payloads.

This module provides business logic schemas that extend the Open Agent Schema Framework (OASF)
with strongly-typed JSON schemas for common business use cases, enabling payload validation
and safe evolution of agent capabilities.
"""
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import jsonschema
from jsonschema import validate, ValidationError
import logging

logger = logging.getLogger(__name__)


class SchemaType(str, Enum):
    """Enumeration of available schema types."""
    ORDER_PROCESSING = "order_processing"
    DATA_ANALYSIS = "data_analysis"
    DOCUMENT_PROCESSING = "document_processing"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    RESEARCH_QUERY = "research_query"


class BusinessLogicSchema(BaseModel):
    """
    Base class for business logic schemas following OASF principles.
    
    This provides a standardized way to define strongly-typed schemas for
    business logic payloads in agent communications.
    """
    schema_version: str = "1.0"
    schema_id: str
    schema_type: SchemaType
    title: str
    description: str
    properties: Dict[str, Any]
    required: List[str] = Field(default_factory=list)
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    
    def to_json_schema(self) -> Dict[str, Any]:
        """
        Convert to JSON Schema format for validation.
        
        Returns:
            JSON Schema dictionary
        """
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": self.schema_id,
            "title": self.title,
            "description": self.description,
            "type": "object",
            "properties": self.properties,
            "required": self.required,
            "examples": self.examples,
            "additionalProperties": False
        }


class OrderProcessingSchema(BusinessLogicSchema):
    """Schema for e-commerce order processing workflows."""
    
    def __init__(self):
        super().__init__(
            schema_id="https://dawn-spec.org/schemas/order-processing/v1.0",
            schema_type=SchemaType.ORDER_PROCESSING,
            title="Order Processing Schema",
            description="Schema for processing e-commerce orders with customer, product, and payment information",
            properties={
                "order_id": {
                    "type": "string",
                    "description": "Unique identifier for the order",
                    "pattern": "^ORD-[A-Z0-9]{8}$"
                },
                "customer_id": {
                    "type": "string",
                    "description": "Unique identifier for the customer",
                    "pattern": "^CUST-[A-Z0-9]{8}$"
                },
                "customer_info": {
                    "type": "object",
                    "description": "Customer information",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "email": {"type": "string", "format": "email"},
                        "phone": {"type": "string", "pattern": "^\\+?[1-9]\\d{1,14}$"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                                "state": {"type": "string"},
                                "postal_code": {"type": "string"},
                                "country": {"type": "string", "minLength": 2, "maxLength": 2}
                            },
                            "required": ["street", "city", "country"]
                        }
                    },
                    "required": ["name", "email"]
                },
                "items": {
                    "type": "array",
                    "description": "List of items in the order",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "string", "pattern": "^PROD-[A-Z0-9]{8}$"},
                            "product_name": {"type": "string", "minLength": 1},
                            "quantity": {"type": "integer", "minimum": 1},
                            "unit_price": {"type": "number", "minimum": 0},
                            "total_price": {"type": "number", "minimum": 0},
                            "category": {"type": "string"},
                            "sku": {"type": "string"}
                        },
                        "required": ["product_id", "product_name", "quantity", "unit_price", "total_price"]
                    }
                },
                "total_amount": {
                    "type": "number",
                    "description": "Total order amount",
                    "minimum": 0
                },
                "currency": {
                    "type": "string",
                    "description": "Currency code",
                    "pattern": "^[A-Z]{3}$",
                    "default": "USD"
                },
                "payment_info": {
                    "type": "object",
                    "description": "Payment information",
                    "properties": {
                        "payment_method": {
                            "type": "string",
                            "enum": ["credit_card", "debit_card", "paypal", "bank_transfer", "cash"]
                        },
                        "payment_status": {
                            "type": "string",
                            "enum": ["pending", "processing", "completed", "failed", "refunded"]
                        },
                        "transaction_id": {"type": "string"}
                    },
                    "required": ["payment_method", "payment_status"]
                },
                "order_status": {
                    "type": "string",
                    "description": "Current order status",
                    "enum": ["pending", "confirmed", "processing", "shipped", "delivered", "cancelled"]
                },
                "created_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Order creation timestamp"
                },
                "updated_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Last update timestamp"
                }
            },
            required=["order_id", "customer_id", "customer_info", "items", "total_amount", "currency", "order_status"],
            examples=[
                {
                    "order_id": "ORD-ABC12345",
                    "customer_id": "CUST-XYZ67890",
                    "customer_info": {
                        "name": "John Doe",
                        "email": "john.doe@example.com",
                        "phone": "+1234567890",
                        "address": {
                            "street": "123 Main St",
                            "city": "Anytown",
                            "state": "CA",
                            "postal_code": "12345",
                            "country": "US"
                        }
                    },
                    "items": [
                        {
                            "product_id": "PROD-DEF11111",
                            "product_name": "Widget Pro",
                            "quantity": 2,
                            "unit_price": 29.99,
                            "total_price": 59.98,
                            "category": "Electronics",
                            "sku": "WGT-PRO-001"
                        }
                    ],
                    "total_amount": 59.98,
                    "currency": "USD",
                    "payment_info": {
                        "payment_method": "credit_card",
                        "payment_status": "completed",
                        "transaction_id": "TXN-123456789"
                    },
                    "order_status": "confirmed",
                    "created_at": "2024-01-15T10:30:00Z",
                    "updated_at": "2024-01-15T10:35:00Z"
                }
            ]
        )


class DataAnalysisSchema(BusinessLogicSchema):
    """Schema for data analysis workflows."""
    
    def __init__(self):
        super().__init__(
            schema_id="https://dawn-spec.org/schemas/data-analysis/v1.0",
            schema_type=SchemaType.DATA_ANALYSIS,
            title="Data Analysis Schema",
            description="Schema for data analysis requests and results",
            properties={
                "analysis_id": {
                    "type": "string",
                    "description": "Unique identifier for the analysis",
                    "pattern": "^ANAL-[A-Z0-9]{8}$"
                },
                "dataset_info": {
                    "type": "object",
                    "description": "Information about the dataset to analyze",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "dataset_name": {"type": "string"},
                        "source": {"type": "string"},
                        "format": {
                            "type": "string",
                            "enum": ["csv", "json", "parquet", "excel", "sql", "api"]
                        },
                        "size_bytes": {"type": "integer", "minimum": 0},
                        "row_count": {"type": "integer", "minimum": 0},
                        "column_count": {"type": "integer", "minimum": 0}
                    },
                    "required": ["dataset_id", "dataset_name", "format"]
                },
                "analysis_type": {
                    "type": "string",
                    "description": "Type of analysis to perform",
                    "enum": [
                        "descriptive", "exploratory", "statistical", "predictive", 
                        "clustering", "classification", "regression", "time_series"
                    ]
                },
                "parameters": {
                    "type": "object",
                    "description": "Analysis-specific parameters",
                    "properties": {
                        "target_column": {"type": "string"},
                        "feature_columns": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "test_size": {"type": "number", "minimum": 0, "maximum": 1},
                        "confidence_level": {"type": "number", "minimum": 0, "maximum": 1},
                        "max_iterations": {"type": "integer", "minimum": 1},
                        "algorithm": {"type": "string"},
                        "hyperparameters": {"type": "object"}
                    }
                },
                "output_format": {
                    "type": "string",
                    "description": "Desired output format",
                    "enum": ["json", "csv", "html", "pdf", "png", "svg"],
                    "default": "json"
                },
                "include_visualizations": {
                    "type": "boolean",
                    "description": "Whether to include data visualizations",
                    "default": False
                },
                "results": {
                    "type": "object",
                    "description": "Analysis results (populated after completion)",
                    "properties": {
                        "summary": {"type": "object"},
                        "metrics": {"type": "object"},
                        "model_performance": {"type": "object"},
                        "visualizations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "title": {"type": "string"},
                                    "data": {"type": "string"},
                                    "format": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            },
            required=["analysis_id", "dataset_info", "analysis_type", "output_format"],
            examples=[
                {
                    "analysis_id": "ANAL-ABC12345",
                    "dataset_info": {
                        "dataset_id": "DS-XYZ67890",
                        "dataset_name": "Sales Data Q4 2023",
                        "source": "company_database",
                        "format": "csv",
                        "size_bytes": 1048576,
                        "row_count": 10000,
                        "column_count": 15
                    },
                    "analysis_type": "descriptive",
                    "parameters": {
                        "target_column": "revenue",
                        "feature_columns": ["product_category", "region", "sales_rep"]
                    },
                    "output_format": "json",
                    "include_visualizations": True
                }
            ]
        )


class DocumentProcessingSchema(BusinessLogicSchema):
    """Schema for document processing workflows."""
    
    def __init__(self):
        super().__init__(
            schema_id="https://dawn-spec.org/schemas/document-processing/v1.0",
            schema_type=SchemaType.DOCUMENT_PROCESSING,
            title="Document Processing Schema",
            description="Schema for document processing, extraction, and transformation tasks",
            properties={
                "task_id": {
                    "type": "string",
                    "description": "Unique identifier for the processing task",
                    "pattern": "^DOC-[A-Z0-9]{8}$"
                },
                "document_info": {
                    "type": "object",
                    "description": "Information about the document to process",
                    "properties": {
                        "document_id": {"type": "string"},
                        "document_name": {"type": "string"},
                        "document_type": {
                            "type": "string",
                            "enum": ["pdf", "docx", "txt", "html", "xml", "csv", "image"]
                        },
                        "size_bytes": {"type": "integer", "minimum": 0},
                        "page_count": {"type": "integer", "minimum": 0},
                        "language": {"type": "string", "minLength": 2, "maxLength": 5},
                        "source_url": {"type": "string", "format": "uri"},
                        "metadata": {"type": "object"}
                    },
                    "required": ["document_id", "document_name", "document_type"]
                },
                "processing_tasks": {
                    "type": "array",
                    "description": "List of processing tasks to perform",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_type": {
                                "type": "string",
                                "enum": [
                                    "text_extraction", "ocr", "entity_extraction", 
                                    "sentiment_analysis", "summarization", "translation",
                                    "classification", "key_phrase_extraction"
                                ]
                            },
                            "parameters": {"type": "object"},
                            "output_format": {"type": "string"}
                        },
                        "required": ["task_type"]
                    }
                },
                "output_options": {
                    "type": "object",
                    "description": "Output formatting options",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["json", "xml", "csv", "txt", "html"],
                            "default": "json"
                        },
                        "include_confidence_scores": {"type": "boolean", "default": True},
                        "include_source_references": {"type": "boolean", "default": False}
                    }
                }
            },
            required=["task_id", "document_info", "processing_tasks"],
            examples=[
                {
                    "task_id": "DOC-ABC12345",
                    "document_info": {
                        "document_id": "PDF-XYZ67890",
                        "document_name": "Contract_Agreement_2024.pdf",
                        "document_type": "pdf",
                        "size_bytes": 2097152,
                        "page_count": 25,
                        "language": "en"
                    },
                    "processing_tasks": [
                        {
                            "task_type": "text_extraction",
                            "output_format": "txt"
                        },
                        {
                            "task_type": "entity_extraction",
                            "parameters": {
                                "entity_types": ["PERSON", "ORG", "DATE", "MONEY"]
                            }
                        }
                    ],
                    "output_options": {
                        "format": "json",
                        "include_confidence_scores": True
                    }
                }
            ]
        )


class ResearchQuerySchema(BusinessLogicSchema):
    """Schema for research query workflows."""
    
    def __init__(self):
        super().__init__(
            schema_id="https://dawn-spec.org/schemas/research-query/v1.0",
            schema_type=SchemaType.RESEARCH_QUERY,
            title="Research Query Schema",
            description="Schema for research queries across multiple data sources",
            properties={
                "query_id": {
                    "type": "string",
                    "description": "Unique identifier for the research query",
                    "pattern": "^RQ-[A-Z0-9]{8}$"
                },
                "query_text": {
                    "type": "string",
                    "description": "The research question or query",
                    "minLength": 10
                },
                "search_domains": {
                    "type": "array",
                    "description": "Domains to search within",
                    "items": {
                        "type": "string",
                        "enum": [
                            "academic_papers", "web_search", "github_repositories",
                            "documentation", "patents", "news_articles", "books"
                        ]
                    },
                    "minItems": 1
                },
                "search_parameters": {
                    "type": "object",
                    "description": "Parameters for the search",
                    "properties": {
                        "max_results_per_domain": {"type": "integer", "minimum": 1, "maximum": 100},
                        "date_range": {
                            "type": "object",
                            "properties": {
                                "start_date": {"type": "string", "format": "date"},
                                "end_date": {"type": "string", "format": "date"}
                            }
                        },
                        "languages": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 2, "maxLength": 5}
                        },
                        "include_citations": {"type": "boolean", "default": True},
                        "quality_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                },
                "synthesis_options": {
                    "type": "object",
                    "description": "Options for synthesizing results",
                    "properties": {
                        "generate_summary": {"type": "boolean", "default": True},
                        "extract_key_concepts": {"type": "boolean", "default": True},
                        "identify_contradictions": {"type": "boolean", "default": False},
                        "output_format": {
                            "type": "string",
                            "enum": ["structured_report", "bullet_points", "narrative", "citations_only"],
                            "default": "structured_report"
                        }
                    }
                }
            },
            required=["query_id", "query_text", "search_domains"],
            examples=[
                {
                    "query_id": "RQ-ABC12345",
                    "query_text": "What are the latest developments in multi-agent systems for distributed AI?",
                    "search_domains": ["academic_papers", "github_repositories", "web_search"],
                    "search_parameters": {
                        "max_results_per_domain": 20,
                        "date_range": {
                            "start_date": "2023-01-01",
                            "end_date": "2024-12-31"
                        },
                        "languages": ["en"],
                        "include_citations": True
                    },
                    "synthesis_options": {
                        "generate_summary": True,
                        "extract_key_concepts": True,
                        "output_format": "structured_report"
                    }
                }
            ]
        )


class SchemaValidator:
    """
    Utility class for validating payloads against business logic schemas.
    
    This class provides methods to validate data against registered schemas
    and return detailed validation results.
    """
    
    def __init__(self):
        """Initialize the schema validator with built-in schemas."""
        self.schemas: Dict[str, BusinessLogicSchema] = {}
        self._register_builtin_schemas()
        
    def _register_builtin_schemas(self) -> None:
        """Register all built-in business logic schemas."""
        schemas = [
            OrderProcessingSchema(),
            DataAnalysisSchema(),
            DocumentProcessingSchema(),
            ResearchQuerySchema()
        ]
        
        for schema in schemas:
            self.register_schema(schema)
            
    def register_schema(self, schema: BusinessLogicSchema) -> None:
        """
        Register a business logic schema.
        
        Args:
            schema: The schema to register
        """
        self.schemas[schema.schema_type.value] = schema
        logger.info(f"Registered schema: {schema.schema_type.value}")
        
    def get_schema(self, schema_type: str) -> Optional[BusinessLogicSchema]:
        """
        Get a registered schema by type.
        
        Args:
            schema_type: The schema type to retrieve
            
        Returns:
            The schema instance or None if not found
        """
        return self.schemas.get(schema_type)
        
    def validate_payload(self, schema_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a payload against a registered schema.
        
        Args:
            schema_type: The type of schema to validate against
            payload: The payload data to validate
            
        Returns:
            Validation result dictionary with success status and details
        """
        schema = self.get_schema(schema_type)
        if schema is None:
            return {
                "valid": False,
                "error": f"Schema type '{schema_type}' not found",
                "schema_type": schema_type
            }
            
        try:
            json_schema = schema.to_json_schema()
            validate(payload, json_schema)
            
            return {
                "valid": True,
                "schema_type": schema_type,
                "schema_version": schema.schema_version,
                "message": "Payload validation successful"
            }
            
        except ValidationError as e:
            return {
                "valid": False,
                "error": str(e.message),
                "error_path": list(e.absolute_path),
                "schema_type": schema_type,
                "invalid_value": e.instance
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}",
                "schema_type": schema_type
            }
            
    def get_validation_errors(self, schema_type: str, payload: Dict[str, Any]) -> List[str]:
        """
        Get detailed validation errors for a payload.
        
        Args:
            schema_type: The type of schema to validate against
            payload: The payload data to validate
            
        Returns:
            List of validation error messages
        """
        result = self.validate_payload(schema_type, payload)
        if result["valid"]:
            return []
        else:
            return [result.get("error", "Unknown validation error")]
            
    def list_available_schemas(self) -> List[Dict[str, Any]]:
        """
        List all available schemas.
        
        Returns:
            List of schema information dictionaries
        """
        schema_list = []
        for schema_type, schema in self.schemas.items():
            schema_list.append({
                "schema_type": schema_type,
                "schema_id": schema.schema_id,
                "title": schema.title,
                "description": schema.description,
                "version": schema.schema_version,
                "required_fields": schema.required,
                "examples_count": len(schema.examples)
            })
        return schema_list
        
    def get_schema_json(self, schema_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the JSON Schema representation of a schema.
        
        Args:
            schema_type: The type of schema to get
            
        Returns:
            JSON Schema dictionary or None if schema not found
        """
        schema = self.get_schema(schema_type)
        if schema:
            return schema.to_json_schema()
        return None


# Global schema validator instance
schema_validator = SchemaValidator()