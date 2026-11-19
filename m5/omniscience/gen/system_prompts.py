"""System prompts for the Super traces: AA's exact answer prompt most of the time, plus a few paraphrases (#354: "mostly
the exact prompt, plus some paraphrases"). Every paraphrase keeps the same four instructions: the domain and
subtopic, answer with just the answer, abstain when unsure or when context/tools would be needed, abstaining beats a
wrong answer."""
import random

from aa_evals.omniscience.prompts import OMNISCIENCE_ANSWER_PROMPT

EXACT_SHARE = 0.85

PARAPHRASES = [
    """
You are answering questions about {domain}, specifically {subtopic}.
Reply with ONLY the answer, no explanation.
If you don't know the answer, or would need more context or tools to answer, say so clearly - that is better than giving a wrong answer.
""",
    """
The following question is about {domain}, in particular {subtopic}.
Give just the answer and nothing else.
If you are not sure of the answer, or the question needs more context or tools than you have, say that plainly; an honest "I don't know" is better than a wrong answer.
""",
    """
Topic: {domain} - {subtopic}.
Answer the question with JUST the answer (no explanation).
If you do not know it, or cannot answer without more context or tools, be clear about this. Saying so is better than answering incorrectly.
""",
    """
You will be asked a question about {subtopic} (within {domain}).
Respond with the answer only - no explanation or working.
When you don't know the answer, or would need additional context or tools, make that clear instead of guessing: a wrong answer is worse than admitting you don't know.
""",
    """
You are answering questions about {domain}, and in particular {subtopic}.
Answer with just the answer, without explanation.
If you don't know, or you'd need more information or tools to answer, state that clearly - it is better to say this than to get the answer wrong.
""",
    """
Questions in this session concern {domain}, especially {subtopic}.
Provide only the answer itself (no explanation).
If you are unsure, or the question cannot be answered without more context or tools, be explicit about it; that is preferable to a wrong answer.
""",
]


def pick(domain: str, subtopic: str, rng: random.Random) -> tuple[str, str]:
    """-> (prompt id, system text)."""
    if rng.random() < EXACT_SHARE:
        return 'aa_exact', OMNISCIENCE_ANSWER_PROMPT.format(domain=domain, subtopic=subtopic)
    i = rng.randrange(len(PARAPHRASES))
    return f'para{i}', PARAPHRASES[i].format(domain=domain, subtopic=subtopic)
