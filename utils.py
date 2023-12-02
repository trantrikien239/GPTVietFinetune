prompt = """Dưới đây là một câu hỏi về sức khoẻ. Hãy viết một câu trả lời phù hợp.

### Câu hỏi:\n{main_question}\n\n### Trả lời:"""

prompt_full = prompt + """\n{answers}"""

def get_text(x):
    return prompt_full.format(
        main_question=x['main_question'], 
        answers=x['answers']
    )