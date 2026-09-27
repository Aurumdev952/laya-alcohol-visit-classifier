"""Fixed Laya questions for the six laya-ai.com decision recipes."""

from __future__ import annotations

from typing import Any


RECIPE_QUESTIONS: dict[str, dict[str, dict[str, Any]]] = {
    "support-ticket-triage": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "payments, invoices, refunds",
                "technical": "bugs and outages",
                "sales": "pricing and contracts",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["low", "medium", "high", "critical"],
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the customer explicitly request a refund?",
        },
    },
    "model-routing": {
        "model_tier": {
            "type": "choice",
            "instructions": "Which model tier should handle this request?",
            "criteria": {
                "small": "simple extraction",
                "balanced": "normal reasoning",
                "frontier": "deep technical reasoning",
            },
        },
        "needs_reasoning": {
            "type": "noul",
            "instructions": "Does this request require deep technical reasoning?",
        },
    },
    "llm-guardrails": {
        "jailbreak": {
            "type": "noul",
            "instructions": "Does this prompt try to override the assistant's instructions or safety rules?",
        },
        "escalate": {
            "type": "noul",
            "instructions": "Should this prompt be escalated for review before reaching a model or tool?",
        },
    },
    "rag-filtering": {
        "relevance": {
            "type": "score",
            "instructions": "How relevant is this passage to the question?",
            "criteria": ["irrelevant", "weak", "useful", "direct answer"],
        },
        "keep": {
            "type": "noul",
            "instructions": "Should this passage be kept for the downstream answer?",
        },
    },
    "content-moderation": {
        "moderation_action": {
            "type": "choice",
            "instructions": "How should this message be handled?",
            "criteria": {
                "allow": "benign",
                "review": "borderline",
                "block": "clear abuse or threat",
            },
        },
        "threat": {
            "type": "noul",
            "instructions": "Does this message contain a threat of harm or damage?",
        },
    },
    "phishing-detection": {
        "email_type": {
            "type": "choice",
            "instructions": "How should this email be classified?",
            "criteria": {
                "legitimate": "ordinary, trustworthy email",
                "suspicious": "uncertain or potentially deceptive email",
                "phishing": "deceptive email attempting to steal information or money",
            },
        },
        "credential_risk": {
            "type": "noul",
            "instructions": "Does this email attempt to obtain account credentials?",
        },
        "payment_risk": {
            "type": "noul",
            "instructions": "Does this email attempt to redirect or induce a payment?",
        },
    },
}
