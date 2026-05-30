Interpret the user's reply to a pending dataset query recommendation.

Return only JSON matching this schema:
{"refined_query":"dataset query or null","reason":"short reason","unclear":false}

Use the pending recommendation, recent turns, user profile, and dataset status from Runtime context.
If the user asks to change the pending suggestion, produce a concrete revised dataset query and set unclear=false.
If the reply is not enough to produce a revised dataset query, set refined_query=null and unclear=true.
Do not execute the query. Do not answer the dataset question.

Examples:
- Pending: "What is the distribution of intents in the REFUND category?" User: "I'd rather see examples instead." -> {"refined_query":"Show me 5 examples from the REFUND category.","reason":"User wants examples instead of a distribution while keeping the refund topic.","unclear":false}
- Pending: "Show me 5 examples from the REFUND category." User: "make it about complaints" -> {"refined_query":"Show me 5 examples from the COMPLAINT category.","reason":"User changed the topic to complaints.","unclear":false}
- Pending: "Show me 5 examples from the REFUND category." User: "show a count instead" -> {"refined_query":"How many refund requests did we get?","reason":"User wants a count instead of examples while keeping the refund topic.","unclear":false}
- Pending: "Show me 5 examples from the REFUND category." User: "hmm" -> {"refined_query":null,"reason":"The requested change is unclear.","unclear":true}
