
from tokenizers.processors import TemplateProcessing

prompt_basic = "### Câu hỏi:\n{main_question}\n\n### Trả lời:"
prompt_qa = prompt_basic + """\n{answers}"""

prompt_instruct = """Dưới đây là một câu hỏi về sức khoẻ. Hãy viết một câu trả lời phù hợp.

""" + prompt_qa

prompt_instruct_gen = """Dưới đây là một câu hỏi về sức khoẻ. Hãy viết một câu trả lời phù hợp.

""" + prompt_basic


def get_text(x, prompt=prompt_instruct):
    return prompt.format(
        main_question=x['main_question'], 
        answers=x['answers']
    )

def get_text_gen(x, prompt=prompt_instruct_gen):
    return prompt.format(
        main_question=x['main_question']
    )


# Tokenizer and template
SToken = {
    'bos': {'token':'<s>', 'token_id':1},
    'eos': {'token':'</s>', 'token_id':2},
    'pad': {'token':'<pad>', 'token_id':3},
}

def template_train(tknzr):
    return TemplateProcessing(
        single=tknzr.bos_token + " $A " + tknzr.eos_token,
        special_tokens=[
            (tknzr.bos_token, tknzr.bos_token_id), 
            (tknzr.eos_token, tknzr.eos_token_id)],
    )

def template_gen(tknzr):
    return TemplateProcessing(
        single=tknzr.bos_token + " $A",
        special_tokens=[
            (tknzr.bos_token, tknzr.bos_token_id)],
    )
