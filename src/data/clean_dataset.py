import numpy as np

try:
    import load_data as data
    from ..models.evaluate_model import DetoxifierEvaluator
    from ..models.toxicity_classifiers.t5_toxicity_evaluator import T5TEModel
    from ..utils.export_utils import export_dataframe
except ImportError:
    import sys
    if '.' not in sys.path: sys.path.append('.')
    import src.data.load_data as data
    from src.models.evaluate_model import DetoxifierEvaluator
    from src.models.toxicity_classifiers.t5_toxicity_evaluator import T5TEModel
    from src.utils.export_utils import export_dataframe

def clean_dataset(input_path, model_path, export_path='data/cleaned/clean.tsv', batch_size=128, verbose=False):
    """
    Given the raw dataset, uses a T5 toxicity regressor to reevaluate whether the text is toxic or not. \
    Then based on the reevaluated dataset, the toxic-toxic, neutral-neutral, and neutral-toxic rows are
    removed, leaving only toxic-neutral rows.
    """
    if verbose: print('Loading data...')
    raw_df = data.load_data(input_path, drop_columns=True, sort_toxicity=True)
    if verbose: print('Loading model...')
    evaluator = DetoxifierEvaluator(T5TEModel(model_path))
    evaluator.set_dataframe(raw_df)
    evaluator.evaluate(batch_size)
    eval_df = evaluator.get_evaluated_dataframe()

    toxic_translations = eval_df['trn_tox']
    neutral_references = np.logical_not(eval_df['ref_tox'])
    untranslated = eval_df['ref_tox'] == eval_df['trn_tox']
    wrong_classed = np.logical_or(toxic_translations, neutral_references)
    to_remove = np.logical_or(untranslated, wrong_classed)

    cleaned_df = eval_df[:][np.logical_not(to_remove)]
    export_dataframe(cleaned_df, export_path)
    if verbose: print(f'The cleaned dataaset is exported to {export_path}')

if __name__ == '__main__':
    clean_dataset('../data/raw/filtered.tsv', 'models/t5-toxicity-regressor/model.pt', verbose=True)
