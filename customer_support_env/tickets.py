"""
Curated ticket corpus with deterministic ground-truth labels.

Each record contains:
  ticket       – the Ticket object fields
  ground_truth – the expected triage output
  history      – revealed when agent calls lookup_history
  extra_info   – revealed when agent calls request_info
"""

TICKET_POOL: dict[str, list[dict]] = {
    # ── TASK 1 ── Priority Classification (easy) ─────────────────────────────
    "task1": [
        {
            "ticket": {
                "ticket_id": "T1001",
                "subject": "URGENT: Unauthorized transactions on my account",
                "body": (
                    "Someone has made unauthorized transactions on my account "
                    "totalling $500 in the last hour. I need this stopped "
                    "immediately and my money returned. This is an emergency!"
                ),
                "customer_tier": "premium",
                "previous_contacts": 0,
                "account_age_days": 730,
                "attachments": ["transaction_screenshot.png"],
            },
            "ground_truth": {
                "priority": "urgent",
                "department": "billing",
                "sentiment": "angry",
                "escalate": True,
            },
            "history": {
                "last_contacts": [],
                "account_flags": ["high_value_customer"],
            },
            "extra_info": {
                "account_status": "active",
                "recent_logins": ["IP: 192.168.1.1 (US)", "IP: 45.33.32.156 (RU)"],
                "note": "Two logins from different countries within 30 min.",
            },
        },
        {
            "ticket": {
                "ticket_id": "T1002",
                "subject": "Dashboard not loading after latest update",
                "body": (
                    "Since your update this morning my main analytics dashboard "
                    "is completely blank. I rely on this for my daily reports. "
                    "Please fix ASAP."
                ),
                "customer_tier": "standard",
                "previous_contacts": 1,
                "account_age_days": 365,
                "attachments": [],
            },
            "ground_truth": {
                "priority": "high",
                "department": "technical",
                "sentiment": "frustrated",
                "escalate": False,
            },
            "history": {
                "last_contacts": [
                    {"date": "2026-03-20", "issue": "slow load times", "resolved": True}
                ],
                "account_flags": [],
            },
            "extra_info": {
                "affected_version": "v4.2.1",
                "known_issue": True,
                "eta_fix": "2026-04-07",
            },
        },
        {
            "ticket": {
                "ticket_id": "T1003",
                "subject": "Question about my upcoming subscription renewal",
                "body": (
                    "Hi, my subscription renews next month and I wanted to check "
                    "whether the price is changing. Could someone let me know? Thanks."
                ),
                "customer_tier": "basic",
                "previous_contacts": 0,
                "account_age_days": 90,
                "attachments": [],
            },
            "ground_truth": {
                "priority": "medium",
                "department": "billing",
                "sentiment": "neutral",
                "escalate": False,
            },
            "history": {"last_contacts": [], "account_flags": []},
            "extra_info": {
                "current_plan": "Basic Monthly $9.99",
                "new_price": "$9.99 (no change)",
            },
        },
        {
            "ticket": {
                "ticket_id": "T1004",
                "subject": "Feature request: dark mode",
                "body": (
                    "Would love to see a dark mode option in the settings. "
                    "Not urgent at all, just a nice-to-have. Love the product!"
                ),
                "customer_tier": "basic",
                "previous_contacts": 0,
                "account_age_days": 45,
                "attachments": [],
            },
            "ground_truth": {
                "priority": "low",
                "department": "general",
                "sentiment": "satisfied",
                "escalate": False,
            },
            "history": {"last_contacts": [], "account_flags": []},
            "extra_info": {"feature_roadmap": "Dark mode is planned for Q3 2026."},
        },
        {
            "ticket": {
                "ticket_id": "T1005",
                "subject": "Cannot log in since yesterday evening",
                "body": (
                    "I've been locked out of my account since yesterday at 6pm. "
                    "Password reset emails are not arriving either. I have a "
                    "deadline tomorrow and really need access."
                ),
                "customer_tier": "premium",
                "previous_contacts": 2,
                "account_age_days": 500,
                "attachments": [],
            },
            "ground_truth": {
                "priority": "high",
                "department": "account",
                "sentiment": "frustrated",
                "escalate": False,
            },
            "history": {
                "last_contacts": [
                    {"date": "2026-04-01", "issue": "password reset", "resolved": False},
                    {"date": "2026-04-02", "issue": "login failure", "resolved": False},
                ],
                "account_flags": ["email_bouncing"],
            },
            "extra_info": {
                "email_status": "bouncing since 2026-04-05",
                "note": "Customer email may be invalid — needs manual verification.",
            },
        },
    ],

    # ── TASK 2 ── Ticket Routing (medium) ────────────────────────────────────
    "task2": [
        {
            "ticket": {
                "ticket_id": "T2001",
                "subject": "Charged twice for the same order",
                "body": (
                    "I was charged $49.99 twice for order #ORD-8821. My bank "
                    "statement shows two identical charges on the same day. "
                    "Please refund the duplicate charge."
                ),
                "customer_tier": "standard",
                "previous_contacts": 0,
                "account_age_days": 200,
                "attachments": ["bank_statement.pdf"],
            },
            "ground_truth": {
                "priority": "high",
                "department": "billing",
                "sentiment": "frustrated",
                "escalate": False,
            },
            "history": {"last_contacts": [], "account_flags": []},
            "extra_info": {
                "order_ORD-8821": "charge processed correctly once; duplicate appears to be a payment gateway error",
            },
        },
        {
            "ticket": {
                "ticket_id": "T2002",
                "subject": "API returning 500 errors — production outage",
                "body": (
                    "Our entire production environment is down. Every call to "
                    "/api/v2/data returns HTTP 500. This is affecting 200+ end "
                    "users right now. We need immediate help."
                ),
                "customer_tier": "premium",
                "previous_contacts": 0,
                "account_age_days": 1095,
                "attachments": ["error_log.txt"],
            },
            "ground_truth": {
                "priority": "urgent",
                "department": "technical",
                "sentiment": "angry",
                "escalate": True,
            },
            "history": {
                "last_contacts": [],
                "account_flags": ["enterprise_sla"],
            },
            "extra_info": {
                "incident_status": "Engineering aware, P1 incident #INC-441 opened",
                "eta": "30 min",
            },
        },
        {
            "ticket": {
                "ticket_id": "T2003",
                "subject": "Return request — item not as described",
                "body": (
                    "I received the wrong colour for item SKU-3392. I'd like to "
                    "return it for a refund. My order was placed 25 days ago."
                ),
                "customer_tier": "basic",
                "previous_contacts": 0,
                "account_age_days": 60,
                "attachments": [],
            },
            "ground_truth": {
                "priority": "medium",
                "department": "returns",
                "sentiment": "neutral",
                "escalate": False,
            },
            "history": {"last_contacts": [], "account_flags": []},
            "extra_info": {
                "return_policy": "30-day return window; 25 days elapsed — within policy",
                "item_SKU-3392": "blue sent, customer ordered red",
            },
        },
        {
            "ticket": {
                "ticket_id": "T2004",
                "subject": "How do I add team members to my workspace?",
                "body": (
                    "I'd like to invite two colleagues but can't find the option "
                    "in settings. Could you point me in the right direction? "
                    "No rush."
                ),
                "customer_tier": "standard",
                "previous_contacts": 0,
                "account_age_days": 30,
                "attachments": [],
            },
            "ground_truth": {
                "priority": "low",
                "department": "general",
                "sentiment": "neutral",
                "escalate": False,
            },
            "history": {"last_contacts": [], "account_flags": []},
            "extra_info": {
                "help_article": "https://docs.example.com/team-invite",
                "note": "Team invite available under Settings > Members",
            },
        },
        {
            "ticket": {
                "ticket_id": "T2005",
                "subject": "Premium features gone after resetting password",
                "body": (
                    "I reset my password yesterday and now all my premium "
                    "features are missing. My subscription is still active and "
                    "I'm still being charged. Please restore my access."
                ),
                "customer_tier": "premium",
                "previous_contacts": 1,
                "account_age_days": 400,
                "attachments": [],
            },
            "ground_truth": {
                "priority": "high",
                "department": "account",
                "sentiment": "frustrated",
                "escalate": False,
            },
            "history": {
                "last_contacts": [
                    {"date": "2026-04-04", "issue": "password reset requested", "resolved": True}
                ],
                "account_flags": ["premium_subscription_active"],
            },
            "extra_info": {
                "subscription_status": "active, paid through 2026-12-01",
                "note": "Password reset may have unlinked OAuth token from subscription record",
            },
        },
    ],

    # ── TASK 3 ── Full Triage (hard) ─────────────────────────────────────────
    "task3": [
        {
            "ticket": {
                "ticket_id": "T3001",
                "subject": "Five years as a customer — this is unacceptable",
                "body": (
                    "I have been a loyal customer for five years and this is the "
                    "third time this month my invoice is wrong. I'm being charged "
                    "for the Enterprise plan but I downgraded to Standard three "
                    "weeks ago. If this isn't fixed today I am cancelling and "
                    "disputing all charges with my bank."
                ),
                "customer_tier": "premium",
                "previous_contacts": 4,
                "account_age_days": 1825,
                "attachments": ["invoices_march.pdf"],
            },
            "ground_truth": {
                "priority": "urgent",
                "department": "billing",
                "sentiment": "angry",
                "escalate": True,
                "response_keywords": ["apologise", "refund", "immediately", "resolve"],
            },
            "history": {
                "last_contacts": [
                    {"date": "2026-03-15", "issue": "invoice overcharge", "resolved": False},
                    {"date": "2026-03-22", "issue": "invoice overcharge follow-up", "resolved": False},
                    {"date": "2026-03-29", "issue": "invoice overcharge — escalation requested", "resolved": False},
                    {"date": "2026-04-02", "issue": "cancellation threat", "resolved": False},
                ],
                "account_flags": ["churn_risk", "high_value_customer", "vip"],
            },
            "extra_info": {
                "subscription_record": "Still shows Enterprise; downgrade request submitted 2026-03-14 but not processed",
                "overpayment": "$60 across 3 billing cycles",
            },
        },
        {
            "ticket": {
                "ticket_id": "T3002",
                "subject": "Integration stopped working — need help urgently",
                "body": (
                    "Our Zapier integration with your platform stopped triggering "
                    "last night. We use it to sync 1,000 records daily for our "
                    "operations team. Everything was fine until your maintenance "
                    "window. Please advise."
                ),
                "customer_tier": "standard",
                "previous_contacts": 0,
                "account_age_days": 280,
                "attachments": ["zapier_error.png"],
            },
            "ground_truth": {
                "priority": "high",
                "department": "technical",
                "sentiment": "frustrated",
                "escalate": False,
                "response_keywords": ["maintenance", "webhook", "investigate", "update"],
            },
            "history": {"last_contacts": [], "account_flags": []},
            "extra_info": {
                "maintenance_log": "Webhook endpoint URL changed from /v1/hook to /v2/hook on 2026-04-05",
                "note": "Customer needs to update Zapier webhook URL — documented in changelog",
            },
        },
        {
            "ticket": {
                "ticket_id": "T3003",
                "subject": "Possible fraudulent return — need guidance",
                "body": (
                    "I received an item that looks used and repackaged. The box "
                    "was sealed but the product inside shows wear. I want a "
                    "replacement or full refund."
                ),
                "customer_tier": "basic",
                "previous_contacts": 0,
                "account_age_days": 15,
                "attachments": ["product_photo1.jpg", "product_photo2.jpg"],
            },
            "ground_truth": {
                "priority": "high",
                "department": "returns",
                "sentiment": "frustrated",
                "escalate": True,
                "response_keywords": ["investigate", "photo", "replacement", "quality"],
            },
            "history": {"last_contacts": [], "account_flags": ["new_account"]},
            "extra_info": {
                "order_details": "Item was returned by a previous customer and incorrectly classified as new",
                "warehouse_flag": "QC failure on re-stocking",
            },
        },
        {
            "ticket": {
                "ticket_id": "T3004",
                "subject": "Need invoice for tax purposes — previous agent didn't help",
                "body": (
                    "I've asked twice before for a VAT invoice for my 2025 annual "
                    "subscription but keep getting told to use the self-service "
                    "portal. The portal only shows 2026 invoices. My accountant "
                    "needs this by end of week."
                ),
                "customer_tier": "standard",
                "previous_contacts": 3,
                "account_age_days": 600,
                "attachments": [],
            },
            "ground_truth": {
                "priority": "high",
                "department": "billing",
                "sentiment": "frustrated",
                "escalate": False,
                "response_keywords": ["invoice", "2025", "email", "apologise"],
            },
            "history": {
                "last_contacts": [
                    {"date": "2026-03-10", "issue": "2025 VAT invoice request", "resolved": False},
                    {"date": "2026-03-20", "issue": "2025 VAT invoice follow-up", "resolved": False},
                    {"date": "2026-04-01", "issue": "2025 VAT invoice — third attempt", "resolved": False},
                ],
                "account_flags": [],
            },
            "extra_info": {
                "invoice_portal_bug": "Pre-2026 invoices not showing in portal — known bug #BUG-902",
                "workaround": "Finance team can email manually — requires internal ticket",
            },
        },
        {
            "ticket": {
                "ticket_id": "T3005",
                "subject": "Account suspended without warning",
                "body": (
                    "My account was suspended this morning with no email, no "
                    "warning, nothing. I can't access any of my data. I run a "
                    "small business on your platform and this is costing me "
                    "money every minute."
                ),
                "customer_tier": "standard",
                "previous_contacts": 0,
                "account_age_days": 120,
                "attachments": [],
            },
            "ground_truth": {
                "priority": "urgent",
                "department": "account",
                "sentiment": "angry",
                "escalate": True,
                "response_keywords": ["suspended", "reinstate", "immediately", "apologise"],
            },
            "history": {"last_contacts": [], "account_flags": ["auto_suspended_payment_failure"]},
            "extra_info": {
                "suspension_reason": "Automatic suspension: payment method declined 3 times",
                "payment_method": "Card ending 4242 expired 2026-03-31",
            },
        },
    ],
}
