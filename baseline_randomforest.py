from modeling.baseline_randomforest import *

if __name__ == '__main__':
    try:
        main()
    except NameError:
        try:
            run_pipeline()
        except NameError:
            try:
                train_and_evaluate()
            except NameError:
                pass
