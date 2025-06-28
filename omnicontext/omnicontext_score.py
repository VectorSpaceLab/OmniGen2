from .prompt_generator import PromptGenerator
from .openai_util import ask_gpt4o
from .json_util import mllm_output_to_dict
import random
import json
import time

class OmniContextScore:
    def __init__(self, openai_url: str, openai_key: str) -> None:
        self.openai_url = openai_url
        self.openai_key = openai_key
        self.prompt_generator = PromptGenerator()

    def evaluate(self, input_image_paths, instruction, with_scene=False):
        results_dict = {}

        max_tries = 3
        PF_scores = None
        SC_scores = None
        for try_idx in range(max_tries):
            try:
                PF_prompt = self.prompt_generator(instruction, task_type="prompt_following")
                SC_prompt = self.prompt_generator(instruction, task_type="subject_consistency", with_scene=with_scene)

                PF_results = ask_gpt4o(input_image_paths, PF_prompt, self.openai_url, self.openai_key)
                SC_results = ask_gpt4o(input_image_paths, SC_prompt, self.openai_url, self.openai_key)

                PF_scores = mllm_output_to_dict(PF_results)
                SC_scores = mllm_output_to_dict(SC_results)

                if PF_scores == "rate_limit_exceeded" or SC_scores == "rate_limit_exceeded":
                    raise Exception("rate_limit_exceeded")
            except Exception as e:
                backoff_time = 2 ** try_idx  # Exponential backoff: 1, 2, 4 seconds
                print(f"{e}, {instruction=}, Attempt {try_idx+1} failed, retrying after {backoff_time} seconds...")
                time.sleep(backoff_time)

        if PF_scores is None:
            guessed_value = random.randint(0, 10)
            print(f"Failed to find the json content in the string for {instruction}. Guess a value : {guessed_value=}.", flush=True)
            PF_scores = {'score': guessed_value, "reasoning": f"guess_if_cannot_parse | {PF_results}"}
        
        if SC_scores is None:
            guessed_value = random.randint(0, 10)
            print(f"Failed to find the json content in the string for {instruction}. Guess a value : {guessed_value=}.", flush=True)
            SC_scores = {'score': guessed_value, "reasoning": f"guess_if_cannot_parse | {SC_results}"}
        
        results_dict["PF_scores"] = PF_scores
        results_dict["SC_scores"] = SC_scores
        return results_dict