import typo
import random
from util import utils, embedding_utils

def char_swap(val):
    myStrErrer = typo.StrErrer(val, seed=random.randint(0, 1000))
    return myStrErrer.char_swap().result

def missing_char(val):
    myStrErrer = typo.StrErrer(val, seed=random.randint(0, 1000))
    return myStrErrer.missing_char().result

def extra_char(val):
    myStrErrer = typo.StrErrer(val, seed=random.randint(0, 1000))
    return myStrErrer.extra_char().result

def nearby_char(val):
    myStrErrer = typo.StrErrer(val, seed=random.randint(0, 1000))
    return myStrErrer.nearby_char().result

def suitable_for_typo_generate(val):
    return len(val) >= 5 and not utils.contain_digit(val) and not utils.contains_non_alphabet(val) and not embedding_utils.is_oov(val)

def generate_one_typo(dist_val):
    filtered_val = [v for v in dist_val if suitable_for_typo_generate(v)]
    if len(filtered_val) == 0: # no typo error can be generated 
        return None
    error_types = [char_swap, extra_char, missing_char, nearby_char]
    val_choice = random.choice(filtered_val)
    error_choice = random.choice(error_types)
    return error_choice(val_choice)

def generate_typo_errors(dist_val, num):
    filtered_val = [v for v in dist_val if suitable_for_typo_generate(v)]
    if len(filtered_val) == 0: # no typo error can be generated 
        return []
    error_types = [char_swap, extra_char, missing_char, nearby_char]
    typos = []
    for _ in range(num):
        val_choice = random.choice(filtered_val)
        error_choice = random.choice(error_types)
        typos.append(error_choice(val_choice))
    return typos