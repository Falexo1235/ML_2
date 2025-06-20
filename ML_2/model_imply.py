import os
import operator
import gc
import re
import logging
from typing import Dict, Optional
from pathlib import Path
import pandas as pd
import torch
from num2words import num2words
from transformers import T5ForConditionalGeneration, GPT2Tokenizer, T5Config
import pickle
import argparse
from tqdm import tqdm
import sys 
import wandb
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data')
MODEL_REPO_PATH = os.path.join(BASE_DIR, "model_repository", "text_normalization", "1")

class TextNormalizationLogger:

    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._setup_logger()
        return cls._instance

    @classmethod
    def _setup_logger(cls):
        log_dir = Path(DATA_PATH)
        log_dir.mkdir(exist_ok=True)

        cls._logger = logging.getLogger('TextNormalization')
        cls._logger.setLevel(logging.INFO)

        for handler in cls._logger.handlers[:]:
            cls._logger.removeHandler(handler)

        file_handler = logging.FileHandler(os.path.join(DATA_PATH, 'log_file.log'), encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        cls._logger.addHandler(file_handler)
        cls._logger.addHandler(console_handler)

    def get_logger(self):
        return self._logger
#rule-baseline
class RuleBasedNormalizer:

    def __init__(self):
        self.logger = TextNormalizationLogger().get_logger()
        self.res_dict = {}
        self.sdict = self._init_substitution_dict()
        #numbers dictionary
        self.SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        self.SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
        self.OTH = str.maketrans("፬", "4")

    def _init_substitution_dict(self) -> Dict[str, str]:
        #special cases dictionary
        sdict = {
            'km2': 'квадратных километров',
            'km²': 'квадратных километров',
            'km': 'километров',
            'kg': 'килограмм',
            'lb': 'фунтов',
            'dr': 'доктор',
            'm²': 'квадратных метров',
            'm2': 'квадратных метров',
            'км2': 'квадратных километров',
            'км²': 'квадратных километров',
            'км': 'километров',
            'кг': 'килограмм',
            'м²': 'квадратных метров',
            'м2': 'квадратных метров',
            '#': 'номер',
            '№': 'номер',
            '%': 'процент',
            '久': 'х_trans и_trans с_trans а_trans',
            '石': 'и_trans с_trans и_trans',
            '譲': 'д_trans з_trans ё_trans',
            '千': 'т_trans и_trans',
            'と': 'т_trans о_trans',
            '尋': 'х_trans и_trans р_trans о_trans',
            'の': 'н_trans о_trans',
            '神': 'к_trans а_trans м_trans и_trans',
            '隠': 'к_trans а_trans к_trans у_trans с_trans и_trans',
            'し': 'с_trans и_trans',
            'イ': 'и_trans',
            'メ': 'м_trans э_trans',
            'ー': '-',
            'ジ': 'д_trans з_trans и_trans',
            'ア': 'а_trans',
            'ル': 'р_trans у_trans',
            'バ': 'б_trans а_trans',
            'ム': 'м_trans у_trans'
        }
        return sdict

    #dictionary training
    def load_training_data(self, train_path: str, data_path: Optional[str] = None) -> bool:
        try:
            self.logger.info(f"Loading training data from {train_path}")

            with open(train_path, encoding='UTF8') as train_file:
                train_file.readline()  
                total = 0
                not_same = 0

                while True:
                    line = train_file.readline().strip()
                    if not line:
                        break

                    total += 1
                    pos = line.find('","')
                    text = line[pos + 2:]
                    if text[:3] == '","':
                        continue

                    text = text[1:-1]
                    arr = text.split('","')
                    if arr[0] != arr[1]:
                        not_same += 1

                    if arr[0] not in self.res_dict:
                        self.res_dict[arr[0]] = {}

                    if arr[1] in self.res_dict[arr[0]]:
                        self.res_dict[arr[0]][arr[1]] += 1
                    else:
                        self.res_dict[arr[0]][arr[1]] = 1

                self.logger.info(f"Main training file: Total: {total}, Different values: {not_same}")

            if data_path and os.path.exists(data_path):
                self._load_additional_data(data_path)

            self.logger.info(f"Training data loaded successfully. Dictionary size: {len(self.res_dict)}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            return False
    #training with additional data from the competition
    def _load_additional_data(self, data_path: str):
        try:
            files = os.listdir(data_path)
            for file in files:
                if file.startswith(".ipynb_checkpoints"):
                    continue

                self.logger.info(f"Reading additional file: {file}")
                file_path = os.path.join(data_path, file)

                with open(file_path, encoding='UTF8') as f:
                    while True:
                        try:
                            line = f.readline().strip()
                            if not line:
                                break

                            pos = line.find('\t')
                            text = line[pos + 1:]
                            if not text:
                                continue

                            arr = text.split('\t')
                            if arr[0] == '<eos>':
                                continue

                            if arr[1] == '<self>' or arr[1] == 'sil':
                                arr[1] = arr[0]

                            if arr[0] not in self.res_dict:
                                self.res_dict[arr[0]] = {}

                            if arr[1] in self.res_dict[arr[0]]:
                                self.res_dict[arr[0]][arr[1]] += 1
                            else:
                                self.res_dict[arr[0]][arr[1]] = 1

                        except Exception as e:
                            self.logger.warning(f"Error processing line in {file}: {e}")
                            continue

                gc.collect()

        except Exception as e:
            self.logger.error(f"Error loading additional data: {e}")

    def normalize_text(self, text: str) -> str:
        try:
            text = text.strip()

            if text in self.res_dict:
                sorted_results = sorted(self.res_dict[text].items(), 
                                      key=operator.itemgetter(1), reverse=True)
                return sorted_results[0][0]

            if len(text) > 1:
                val = text.split(',')
                if len(val) == 2 and val[0].isdigit() and val[1].isdigit():
                    text = ''.join(val)

            if text.isdigit():
                normalized = text.translate(self.SUB)
                normalized = normalized.translate(self.SUP)
                normalized = normalized.translate(self.OTH)
                try:
                    return num2words(float(normalized), lang='ru')
                except:
                    return text

            elif len(text.split(' ')) > 1:
                val = text.split(' ')
                for i, v in enumerate(val):
                    if v.isdigit():
                        normalized = v.translate(self.SUB)
                        normalized = normalized.translate(self.SUP)
                        normalized = normalized.translate(self.OTH)
                        try:
                            val[i] = num2words(float(normalized), lang='ru')
                        except:
                            pass
                    elif v in self.sdict:
                        val[i] = self.sdict[v]

                return ' '.join(val)

            elif text in self.sdict:
                return self.sdict[text]

            return text

        except Exception as e:
            self.logger.error(f"Error in rule-based normalization for '{text}': {e}")
            return text
    #dates fix from competition comments
    def apply_date_fixes(self, text: str) -> str:
        replacements = {
            'two thousand and sixteen': 'две тысячи шестнадцатого',
            'two thousand and seventeen': 'две тысячи семнадцатого',
            'eleven': 'одиннадцатого',
            'twelve': 'двенадцатого',
            'thirteen': 'тринадцатого',
            'fourteen': 'четырнадцатого',
            'fifteen': 'пятнадцатого',
            'sixteen': 'шестнадцатого',
            'seventeen': 'семнадцатого',
            'eighteen': 'восемнадцатое',
            'nineteen': 'девятнадцатого',
            'twenty-one': 'двадцать первого',
            'twenty-two': 'двадцать второго',
            'twenty-three': 'двадцать третьего',
            'twenty-four': 'двадцать четвертого',
            'twenty-five': 'двадцать пятого',
            'twenty-six': 'двадцать шестого',
            'twenty-seven': 'двадцать седьмого',
            'twenty-eight': 'двадцать восьмого',
            'twenty-nine': 'двадцать девятого',
            'twenty': 'двадцатого',
            'thirty-one': 'тридцать первого',
            'thirty': 'тридцатого',
            'two': 'второго',
            'three': 'третьего',
            'four': 'четвертого',
            'five': 'пятого',
            'six': 'шестого',
            'seven': 'седьмого',
            'eight': 'восьмого',
            'nine': 'девятого',
            'one': 'первого',
            'ten': 'десятого'
        }

        result = text
        for eng, rus in replacements.items():
            result = result.replace(eng, rus)
        return result
#neural text normalization
class SaarusTextNormalizer:

    def __init__(self):
        self.logger = TextNormalizationLogger().get_logger()
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cpu')
        self.re_tokens = re.compile(r"(?:[.,!?]|[а-яА-Я]\S*|\d\S*(?:.\d+)?|[^а-яА-Я\d\s]+)\s*")
        #roman numbers dictionary for special cases
        self.roman_numerals = {
            'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
            'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
            'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX',
            'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM', 'M'
        }
    #exporting the model
    def save_pytorch_model(self, save_dir: str):
        # Save the full model (not just state_dict) in the format expected by from_pretrained
        self.model.save_pretrained(save_dir)
        
        if hasattr(self.model, 'config'):
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                f.write(self.model.config.to_json_string())
    #importing the model
    def load_pytorch_model(self, model_dir: str) -> bool:
        try:
            # Load the model directly using from_pretrained
            self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()

            self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

            self.logger.info("PyTorch model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error loading PyTorch model: {e}")
            return False

    def load_model(self) -> bool:
        try:
            self.logger.info("Loading saarus72/russian_text_normalizer model...")

            self.tokenizer = GPT2Tokenizer.from_pretrained(
                "saarus72/russian_text_normalizer", 
                eos_token='</s>'
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                "saarus72/russian_text_normalizer"
            )
            self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"Model loaded successfully on {self.device}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading neural model: {e}")
            return False

    def is_roman_numeral(self, text: str) -> bool:
        text_upper = text.upper().strip()
        return text_upper in self.roman_numerals

    def tokenize(self, text: str):
        return re.findall(self.re_tokens, text)

    def strip_numbers(self, s: str) -> str:
        result = []
        for part in s.split():
            if part.isdigit():
                while len(part) > 3:
                    result.append(part[:- 3 * ((len(part) - 1) // 3)])
                    part = part[- 3 * ((len(part) - 1) // 3):]
                if part:
                    result.append(part)
            else:
                result.append(part)
        return " ".join(result)

    def construct_prompt(self, text: str) -> str:
        result = "<SC1>"
        etid = 0
        token_to_add = ""

        for token in self.tokenize(text) + [""]:
            if not re.search("[a-zA-Z\d]", token):
                if token_to_add:
                    end_match = re.search(r"(.+?)(\W*)$", token_to_add, re.M)
                    if end_match:
                        groups = end_match.groups()
                        result += f"[{self.strip_numbers(groups[0])}]<extra_id_{etid}>{groups[1]}"
                        etid += 1
                        token_to_add = ""
                result += token
            else:
                token_to_add += token
        return result
    def predict(self, text: str) -> str:
        try:
            input_ids = torch.tensor([self.tokenizer.encode(text)]).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids, 
                    max_new_tokens=50, 
                    eos_token_id=self.tokenizer.eos_token_id, 
                    early_stopping=True,
                    do_sample=False
                )

            return self.tokenizer.decode(outputs[0][1:])

        except Exception as e:
            self.logger.error(f"Error in model prediction for '{text}': {e}")
            return ""

    def construct_answer(self, prompt: str, prediction: str) -> str:
        re_prompt = re.compile(r"\[([^\]]+)\]<extra_id_(\d+)>")
        re_pred = re.compile(r"\<extra_id_(\d+)\>(.+?)(?=\<extra_id_\d+\>|</s>)")

        pred_data = {}
        for match in re.finditer(re_pred, prediction.replace("\n", " ")):
            pred_data[match[1]] = match[2].strip()

        while match := re.search(re_prompt, prompt):
            replace = pred_data.get(match[2], match[1])
            prompt = prompt[:match.span()[0]] + replace + prompt[match.span()[1]:]

        return prompt.replace("<SC1>", "")

    def normalize_text(self, text: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            prompt = self.construct_prompt(text)
            start_time = time.time()
            prediction = self.predict(prompt)
            answer = self.construct_answer(prompt, prediction)
        
            if hasattr(self, "use_wandb") and self.use_wandb:
                processing_time = time.time() - start_time
                wandb.log({
                    "neural_processing_time": processing_time,
                    "text_length": len(text)
                })
            return answer.strip()

        except Exception as e:
            self.logger.error(f"Error in neural normalization for '{text}': {e}")
            return text

class My_TextNormalization_Model:
    def __init__(self):
        self.logger = TextNormalizationLogger().get_logger()
        self.rule_based = RuleBasedNormalizer()
        self.neural = SaarusTextNormalizer()
        self.neural_loaded = False
        
        api_key = os.getenv('WANDB_API_KEY')
        wandb.login(key=api_key)
        wandb.init(
            project="text-normalization",
            config={
                "model":"Rule-based + Saarus T5",
                "framework":"PyTorch",
                "rule_based_dict_size":0
            }
        )
        self.use_wandb = True

        self.logger.info("Text Normalization Model initialized")

    def save_models(self, save_dir: str):
        try:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            with open(os.path.join(save_dir, 'rule_based_dict.pkl'), 'wb') as f:
                pickle.dump(self.rule_based.res_dict, f)

            wandb.log({"rule_based_dict_size": len(self.rule_based.res_dict)})
            model_artifact = wandb.Artifact(
                    name="text_normalization_model",
                    type="model",
                    description="Rule-based dictionary and neural model weights"
                )
            model_artifact.add_dir(save_dir)
            wandb.log_artifact(model_artifact)
            examples=list(self.rule_based.res_dict.items())[:10]
            table = wandb.Table(columns=["Original","Normalized","Count"])
            for original, normalizations in examples:
                normalized, count = max(normalizations.items(), key=lambda x: x[1])
                table.add_data(original,normalized,count)
            wandb.log({"rule_based_examples": table})
            
            self.neural.save_pytorch_model(save_dir)

            self.neural.tokenizer.save_pretrained(save_dir)

            self.logger.info(f"Models saved successfully to {save_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            return False
        
    def load_saved_models(self, model_dir: str) -> bool:
        try:

            rule_dict_path = os.path.join(model_dir, 'rule_based_dict.pkl')
            if os.path.exists(rule_dict_path):
                with open(rule_dict_path, 'rb') as f:
                    self.rule_based.res_dict = pickle.load(f)

            self.neural_loaded = self.neural.load_pytorch_model(model_dir)

            self.logger.info("Models loaded successfully from saved state")
            return True

        except Exception as e:
            self.logger.error(f"Error loading saved models: {e}")
            return False

    def load_model(self, train_path: Optional[str] = None, data_path: Optional[str] = None) -> bool:
        try:
            self.logger.info("Loading text normalization models...")

            if train_path:
                if not self.rule_based.load_training_data(train_path, data_path):
                    self.logger.warning("Failed to load rule-based training data")

            self.neural_loaded = self.neural.load_model()
            if not self.neural_loaded:
                self.logger.warning("Neural model not loaded, will use rule-based only")

            self.logger.info("Model loading completed")
            return True

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False

    def normalize_text(self, text: str) -> str:
        try:
            if not text or not text.strip():
                return text

            text = text.strip()
            self.logger.debug(f"Normalizing text: '{text}'")

            rule_result = self.rule_based.normalize_text(text)

            if self._needs_neural_processing(text, rule_result):
                if self.neural_loaded:
                    try:
                        #neural_result = self.neural.normalize_text(text)
                        final_result = self._postprocess_neural_result(text, neural_result)
                        self.logger.debug(f"Neural normalization: '{text}' -> '{final_result}'")
                        return final_result
                    except Exception as e:
                        self.logger.warning(f"Neural processing failed for '{text}': {e}")
                        return rule_result
                else:
                    self.logger.debug(f"Neural model not available, using rule-based result")
                    return rule_result

            final_result = self.rule_based.apply_date_fixes(rule_result)
            self.logger.debug(f"Rule-based normalization: '{text}' -> '{final_result}'")
            return final_result

        except Exception as e:
            self.logger.error(f"Error normalizing text '{text}': {e}")
            return text

    def _postprocess_neural_result(self, original_text: str, neural_result: str) -> str:
        original_clean = original_text.strip()
        neural_clean = neural_result.strip()

        if (re.match(r'^[а-яА-Я\s]+$', neural_clean) and 
            not re.search(r'\d', original_clean) and 
            not re.search(r'[а-яА-Я]', original_clean) and
            not self.neural.is_roman_numeral(original_clean)):

            letters = []
            for char in neural_clean.replace(' ', ''):
                if char.isalpha():
                    letters.append(f"{char.lower()}_trans")
                else:
                    letters.append(char)
            return ' '.join(letters)

        return neural_clean

    def process_file(self, input_path: str, output_path: str) -> bool:
        try:
            self.logger.info(f"Processing file: {input_path}")
            df = pd.read_csv(input_path)
            total_rows = len(df)
            self.logger.info(f"Loaded {total_rows} rows from {input_path}")

            main_progress = tqdm(
                total=total_rows,
                desc="Processing file",
                position=0,
                file=sys.stdout,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )

            neural_texts = []
            neural_indices = []
            results = []

            for i, row in enumerate(df.itertuples()):
                try:
                    original_text = str(row.before).strip()
                    composite_id = f"{row.sentence_id}_{row.token_id}"

                    rule_result = self.rule_based.normalize_text(original_text)
                    final_result = self.rule_based.apply_date_fixes(rule_result)

                    results.append({
                        'id': composite_id,
                        'after': final_result
                    })

                    if self._needs_neural_processing(original_text, final_result):
                        neural_texts.append(original_text)
                        neural_indices.append(i)

                    main_progress.update(1)

                except Exception as e:
                    self.logger.error(f"Error processing row {i}: {e}")
                    results.append({
                        'id': f"{getattr(row, 'sentence_id', 'unknown')}_{getattr(row, 'token_id', 'unknown')}",
                        'after': str(getattr(row, 'before', ''))
                    })
                    main_progress.update(1)

            main_progress.close()

            if neural_texts:
                neural_progress = tqdm(
                    total=len(neural_texts),
                    desc="Neural processing",
                    position=0,
                    file=sys.stdout,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                )

                for idx, text in zip(neural_indices, neural_texts):
                    try:
                        neural_result = self.neural.normalize_text(text)
                        final_result = self._postprocess_neural_result(text, neural_result)
                        results[idx]['after'] = final_result
                    except Exception as e:
                        self.logger.warning(f"Neural processing failed for '{text}': {e}")

                    neural_progress.update(1)

                neural_progress.close()

            # Log to wandb after processing is complete
            wandb.config.update({
                "input_file": input_path,
                "output_file": output_path,
                "total_rows": total_rows
            })
            
            # Log sample results
            sample_results = []
            for i in range(min(100, len(results))):
                # Get original text from the dataframe for logging
                original_for_logging = str(df.iloc[i]['before']) if i < len(df) else ''
                sample_results.append([
                    results[i]['id'],
                    original_for_logging,
                    results[i]['after']
                ])
            results_table = wandb.Table(
                columns=["ID", "Original", "Normalized"],
                data=sample_results
            )
            wandb.log({"sample_results": results_table})

            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            self.logger.info(f"Results saved to {output_path}")

            neural_count = len(neural_texts)
            self.logger.info(
                f"Processing completed: {total_rows} total rows, "
                f"{neural_count} neural processed ({neural_count/total_rows:.1%})"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error processing file: {e}")
            return False

    def _needs_neural_processing(self, original_text: str, rule_result: str) -> bool:
        if "_trans" in rule_result:
            return False

        if self.neural.is_roman_numeral(original_text):
            return False

        if original_text == rule_result:
            # Check if text contains digits or specific English number words
            return bool(
                re.search(r'\d', original_text) or 
                re.search(r'thousand|hundred|fifty|million|billion', original_text, re.IGNORECASE)
            )

        return False
    def __del__(self):
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='Text Normalization System')
    parser.add_argument('--train', action='store_true', help='Train and save models')
    parser.add_argument('--normalize', action='store_true', help='Run text normalization')
    parser.add_argument('--input', type=str, help='Input file path for normalization', default=os.path.join(DATA_PATH, 'ru_test_2.csv'))
    parser.add_argument('--output', type=str, help='Output file path for results', default=os.path.join(DATA_PATH, 'final_submission.csv'))
    args = parser.parse_args()

    model = My_TextNormalization_Model()

    if args.train:
        wandb.config.update({
            "training_data": os.path.join(DATA_PATH, 'ru_train.csv'),
            "additional_data": os.path.join(DATA_PATH, 'ru_with_types')
        })
        model.load_model(
            train_path=os.path.join(DATA_PATH, 'ru_train.csv'),
            data_path=os.path.join(DATA_PATH, 'ru_with_types')
        )
        model.save_models(MODEL_REPO_PATH)

    elif args.normalize and args.input and args.output:

        model.load_saved_models(MODEL_REPO_PATH)
        model.process_file(args.input, args.output)

if __name__ == "__main__":
    main()
