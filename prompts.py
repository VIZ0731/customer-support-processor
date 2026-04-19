# prompts.py (UTF-8 encoded)
# All prompt templates for the customer support processor

# ========== Step 1: Preprocessing (clean message) ==========
PREPROCESS_PROMPT = """
You are a text preprocessing assistant. Please clean and normalize the following user message:

1. Fix spelling errors
2. Expand abbreviations (e.g., "u" -> "you", "pls" -> "please")
3. Standardize format

Original user message: {raw_message}

Return only the cleaned text, no explanations.
"""

# ========== Step 2: Classification ==========
CLASSIFICATION_PROMPT = """
You are a ticket classification assistant. Analyze the following user message and determine the ticket category and extract key information.

User message: {cleaned_message}

Return a JSON object with the following fields:
- category: one of TECHNICAL, BILLING, GENERAL, COMPLAINT
- product_name: product name if mentioned
- issue_type: type of issue
- urgency: low/medium/high

Return only JSON, no other text.

Example output:
{{"category": "TECHNICAL", "product_name": "Mobile App", "issue_type": "login_failure", "urgency": "high"}}
"""

# ========== Step 3: Branch-specific response generation ==========

TECHNICAL_PROMPT = """
You are a technical support expert. The user has a technical issue. Generate a professional, helpful response.

Classification info: {classification}
User message: {message}

Your response should include:
1. Empathy and understanding of the issue
2. Concrete troubleshooting steps (3-4 steps)
3. Instructions for further contact if issue persists

Return only the response content.
"""

BILLING_PROMPT = """
You are a billing support specialist. The user has a billing or refund related question. Generate a clear, professional response.

Classification info: {classification}
User message: {message}

Your response should include:
1. Confirmation of the user's issue
2. Clear policy explanation or next steps
3. Expected processing time
4. Contact information for further assistance

Return only the response content.
"""

GENERAL_PROMPT = """
You are a customer service assistant. The user has a general inquiry. Generate a friendly, informative response.

Classification info: {classification}
User message: {message}

Your response should include:
1. Direct answer to the question
2. Relevant helpful information or links
3. Invitation for further help

Return only the response content.
"""

COMPLAINT_PROMPT = """
You are a customer service supervisor. The user has expressed dissatisfaction. Generate a sincere, empathetic response.

Classification info: {classification}
User message: {message}

Your response should include:
1. Sincere apology and empathy
2. Specific remedy actions
3. Explanation of escalation process
4. Clear follow-up commitment

Return only the response content.
"""

# ========== Parallel Task: Sentiment Analysis ==========
SENTIMENT_PROMPT = """
Analyze the sentiment of the following user message.

User message: {message}

Return JSON with:
- sentiment: positive/neutral/negative
- confidence: number between 0 and 1
- brief_reason: short reason

Return only JSON.
"""

# ========== Parallel Task: Keyword Extraction ==========
KEYWORD_PROMPT = """
Extract key entities and keywords from the following user message.

User message: {message}

Return JSON with:
- keywords: list of 3-5 keywords
- entities: list of entities (product names, person names, numbers, etc.)

Return only JSON.
"""

# ========== Reflection: Evaluation ==========
EVALUATION_PROMPT = """
You are a quality evaluation expert. Evaluate the following customer service response.

Original user message: {message}
Classification info: {classification}
Generated response: {response}

Evaluate on a scale of 1-10 for:
1. Tone appropriateness: friendly, professional, empathetic?
2. Completeness: covers all necessary information?
3. Accuracy: information is accurate and relevant?

Return JSON:
{{
    "tone_score": 8,
    "completeness_score": 7,
    "accuracy_score": 9,
    "total_score": 8.0,
    "critique": "specific critique and suggestions",
    "needs_improvement": true/false
}}
"""

# ========== Reflection: Improvement ==========
IMPROVEMENT_PROMPT = """
You are a response improvement expert. Improve the response based on the evaluation feedback.

Original response: {response}
Evaluation critique: {critique}
User message: {message}
Classification info: {classification}

Return the improved response. Return only the response, no explanations.
"""