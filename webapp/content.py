"""Plain-language teaching text for the Model Insights and About pages.

Written for a student who is new to machine learning: short sentences, one idea at a time, and an
everyday comparison for each idea. Nothing in here is an experimental result.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Concepts: (title, simple explanation, everyday comparison, how it is used in this project)
# ---------------------------------------------------------------------------
CONCEPTS = [
    {
        "title": "Sentiment analysis",
        "simple": (
            "Sentiment analysis means teaching a computer to read a piece of text and decide how the writer feels. "
            "In this project the computer reads one product review and answers with one of three labels: "
            "**negative**, **neutral** or **positive**."
        ),
        "analogy": "It is like a shop assistant who reads a feedback card and puts it in the 'unhappy', 'okay' or 'happy' pile.",
        "project": (
            "The data has star ratings, not feelings. We turn stars into labels: 1-2 stars = negative, "
            "3 stars = neutral, 4-5 stars = positive."
        ),
    },
    {
        "title": "TF-IDF (turning text into numbers)",
        "simple": (
            "Computers cannot learn from words directly, so we turn each review into a list of numbers. "
            "TF-IDF gives every word a score in every review. **TF** (term frequency) asks: how often does this "
            "word appear in *this* review? **IDF** (inverse document frequency) asks: how *rare* is this word across "
            "*all* reviews? A word that is frequent here but rare everywhere else gets a high score. Very common "
            "words such as 'the' get a low score."
        ),
        "analogy": "A word like 'itchy' on a clothing review tells you a lot. A word like 'the' tells you nothing, because everyone uses it.",
        "project": (
            "We use single words and word pairs (for example 'not good'), ignore words seen in only one review, "
            "and keep words like 'not' and 'no' because they flip the meaning of a sentence. "
            "The TF-IDF vocabulary is learned from the **training data only**, so no test information leaks in."
        ),
    },
    {
        "title": "Logistic Regression",
        "simple": (
            "Logistic Regression gives every word a **weight** for each class. A word such as 'love' gets a big weight "
            "for *positive*; a word such as 'returned' gets a big weight for *negative*. To judge a review it adds up "
            "the weights of its words and turns the totals into percentages (probabilities) that add up to 100%."
        ),
        "analogy": "Each word casts votes for a class; the class with the most votes wins, and the vote shares become the percentages.",
        "project": "We tried six strengths of regularisation (C) with and without class weighting and kept the best one on the validation data.",
    },
    {
        "title": "Multinomial Naive Bayes",
        "simple": (
            "Naive Bayes counts how often each word appears in negative, neutral and positive reviews. For a new review "
            "it asks: which class makes these words most likely? It is called 'naive' because it pretends that words "
            "do not affect each other, which is not true, but it still works surprisingly well and is extremely fast."
        ),
        "analogy": "If a card says 'broken', 'refund' and 'late', you guess 'unhappy' because those words were mostly seen on unhappy cards.",
        "project": "Its smoothing value (alpha) and whether to assume equal class sizes were chosen on the validation data.",
    },
    {
        "title": "Linear SVM (Support Vector Machine)",
        "simple": (
            "A Linear SVM looks for the best straight dividing line (in a space with one direction per word) that keeps "
            "the classes as far apart as possible. The review lands on one side of the line. "
            "Its output is a **distance from the line**, not a probability, so we never show it as a percentage."
        ),
        "analogy": "Imagine pushing a wide road between two crowds of points: the SVM picks the widest road that fits.",
        "project": "We tried five values of C with and without class weighting and kept the best on validation data.",
    },
    {
        "title": "Transformers",
        "simple": (
            "A transformer reads **all the words of a sentence together** and uses a trick called *attention* to let each "
            "word look at the other words. That is how it knows that 'not' changes 'good', or that 'cold' means something "
            "different in 'cold coffee' and 'cold dress'. Transformers first learn English by reading huge amounts of text."
        ),
        "analogy": "TF-IDF models read a bag of words. A transformer reads the sentence the way a person does, in order and in context.",
        "project": "We use a transformer called DistilBERT, fine-tuned for our three labels.",
    },
    {
        "title": "DistilBERT",
        "simple": (
            "DistilBERT is a smaller, faster copy of the famous BERT model. It was made by *knowledge distillation*: "
            "a small 'student' model learns to imitate a big 'teacher' model. It has about 66 million adjustable numbers "
            "(parameters) and 6 layers. The DistilBERT paper (Sanh et al., 2019) reports that it is about 40% smaller and "
            "60% faster than BERT while keeping most of its language understanding."
        ),
        "analogy": "A student who studied the teacher's answers can do most of the teacher's job with much less effort.",
        "project": "We use the 'distilbert-base-uncased' version (it ignores capital letters).",
    },
    {
        "title": "Fine-tuning",
        "simple": (
            "Fine-tuning means taking a model that already understands English and training it a little more on our "
            "own task. We add a small new output layer with 3 outputs (negative, neutral, positive) and then train the "
            "whole network on our reviews for a few passes (epochs). After every epoch we check the validation score and "
            "keep the best version."
        ),
        "analogy": "Hiring someone who already speaks English and giving them one week of training on your shop's reviews, instead of teaching them English from zero.",
        "project": "Learning rate and class weighting were compared on the validation data; the test data was not used.",
    },
]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
METRICS = [
    {
        "title": "Accuracy",
        "formula": "correct predictions / all predictions",
        "simple": "Out of 100 reviews, how many did the model label correctly? Easy to understand, but it can fool you: "
        "if 77 of 100 reviews are positive, a model that always says 'positive' scores 77% without learning anything.",
    },
    {
        "title": "Precision",
        "formula": "correct 'neutral' predictions / all 'neutral' predictions",
        "simple": "When the model says 'neutral', how often is it right? High precision means few false alarms.",
    },
    {
        "title": "Recall",
        "formula": "correct 'neutral' predictions / all truly neutral reviews",
        "simple": "Of all the truly neutral reviews, how many did the model find? High recall means few misses.",
    },
    {
        "title": "F1-score",
        "formula": "2 x precision x recall / (precision + recall)",
        "simple": "One number that is high only when precision AND recall are both high. It is calculated for each class.",
    },
    {
        "title": "Macro F1-score",
        "formula": "average of the F1 of negative, neutral and positive",
        "simple": "The plain average of the three class F1-scores, so every class counts equally. Our data is mostly positive, "
        "so this is the fairest single number: a model cannot score well by ignoring the small neutral and negative classes. "
        "It is the number we use to pick settings and to rank models.",
    },
    {
        "title": "Confusion matrix",
        "formula": "rows = true class, columns = predicted class",
        "simple": "A small table that shows exactly where the model gets confused. The diagonal is correct answers. "
        "Numbers off the diagonal are mistakes, for example a neutral review predicted as positive.",
    },
    {
        "title": "Training time and inference time",
        "formula": "seconds to learn / milliseconds to answer",
        "simple": "**Training time** is how long the model takes to learn from the training data (once). "
        "**Inference time** is how long it takes to label one new review (every time it is used). "
        "A model can be very accurate but too slow or too big for some uses, so we measure both.",
    },
]

# ---------------------------------------------------------------------------
# Classical ML versus transformer (qualitative; real measured numbers are on the Model Comparison page)
# ---------------------------------------------------------------------------
COMPARISON_ROWS = [
    ("How it reads text", "Counts words and word pairs (TF-IDF). Order and context are mostly ignored.",
     "Reads the whole sentence with attention, so word order and context matter."),
    ("Prior knowledge", "Starts from zero and learns only from our reviews.",
     "Starts with general English knowledge from pre-training, then is fine-tuned."),
    ("Training hardware", "A normal laptop CPU is enough.", "A GPU is strongly recommended."),
    ("Training time", "Seconds to minutes.", "Minutes to hours."),
    ("Model size", "A few megabytes.", "Hundreds of megabytes."),
    ("Speed per review", "Very fast (about a millisecond or less).", "Slower; needs much more computation."),
    ("Explaining a decision", "Easy: you can list the words with the biggest weights.", "Hard: millions of numbers interact."),
    ("Typical strength", "Cheap, fast, transparent baseline.", "Better at subtle language such as mixed or neutral reviews."),
]

# ---------------------------------------------------------------------------
# Project brief
# ---------------------------------------------------------------------------
PROBLEM_STATEMENT = (
    "Online shops receive thousands of customer reviews. Reading them by hand is too slow, so shops want software that "
    "can label each review as negative, neutral or positive. There are two very different ways to build such software: "
    "simple, fast classical machine learning (TF-IDF with Logistic Regression, Naive Bayes or a Linear SVM) and a modern "
    "transformer (DistilBERT). This project asks: **how much better is the transformer, and is the extra cost in "
    "training time, model size and speed worth it?**"
)

OBJECTIVES = [
    "Build a clean, reproducible dataset of e-commerce reviews with three sentiment labels.",
    "Train and tune three classical models (Logistic Regression, Naive Bayes, Linear SVM) on TF-IDF features.",
    "Fine-tune DistilBERT for the same three labels.",
    "Compare all four models fairly on one untouched test set, using macro F1 as the main number.",
    "Measure the cost of each model: training time, inference time, model size and number of parameters.",
    "Present everything in an interactive dashboard that anyone can try on their own review text.",
]

METHODOLOGY = [
    ("1. Data", "Women's E-Commerce Clothing Reviews (Kaggle, CC0). Remove rows with no text, map stars to labels, remove duplicate reviews."),
    ("2. Split", "Stratified 70% train / 15% validation / 15% test with a fixed random seed, so every model sees exactly the same data."),
    ("3. Classical models", "TF-IDF is fitted on the training split only. Settings of each model are chosen on the validation split using macro F1."),
    ("4. DistilBERT", "Fine-tuned on the training split with a plain PyTorch loop. Learning rate and class weighting are compared on the validation split; the best epoch is kept."),
    ("5. Evaluation", "Every model is scored once on the test split. We report accuracy, macro precision, macro recall, macro F1, per-class scores and confusion matrices."),
    ("6. Cost", "Training time, inference time per review, model size and parameter count are measured for every model."),
    ("7. Dashboard", "This Streamlit app loads the saved models and result files. It never trains anything."),
]

LIMITATIONS = [
    "**Labels come from star ratings, not from reading the text.** A 3-star review may be mostly positive, a 4-star review may "
    "contain complaints, and different people use stars differently. Some 'errors' may really be label noise.",
    "**The classes are unbalanced.** About 77% of the reviews are positive, so accuracy alone is misleading. Neutral is the hardest "
    "class because its wording overlaps with both other classes.",
    "**One domain, one language.** All reviews are about women's clothing and are in English. The results may not carry over to "
    "electronics, food or other languages.",
    "**One test split, one run.** We report single numbers without confidence intervals. Small differences between models "
    "(a point or two of macro F1) may not be real.",
    "**Limited tuning.** Only a small grid of settings was tried for each model, and DistilBERT was trained on a laptop GPU.",
    "**Training time depends on the hardware.** Classical models were timed on the CPU and DistilBERT on the GPU used for training, "
    "so the times show the order of magnitude, not an exact like-for-like benchmark.",
    "**Neither model explains itself fully.** Sarcasm, humour and very short reviews are hard for both families.",
]

FUTURE_SCOPE = [
    "Use cross-validation or bootstrapping to give confidence intervals for every score.",
    "Repeat the study on other review datasets (for example Amazon or Flipkart reviews) to test whether the findings hold.",
    "Try larger transformers (BERT, RoBERTa) and compare the gain with the extra cost.",
    "Treat the 1-5 stars as an ordered scale instead of three separate classes.",
    "Handle class imbalance with other techniques (resampling, focal loss) and calibrate the probabilities.",
    "Add explanations for single predictions (for example LIME or SHAP).",
    "Support mixed-language reviews and package the best model as a small web API.",
]

TECH_STACK = [
    ("Language", "Python"),
    ("Classical ML", "scikit-learn (TF-IDF, Logistic Regression, Multinomial Naive Bayes, Linear SVM)"),
    ("Deep learning", "PyTorch"),
    ("Transformers", "Hugging Face Transformers (distilbert-base-uncased)"),
    ("Data", "pandas, NumPy"),
    ("Charts", "Plotly (dashboard), Matplotlib (saved figures)"),
    ("Dashboard", "Streamlit"),
    ("Tests", "pytest and Streamlit AppTest"),
]
