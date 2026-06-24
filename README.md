# song-popularity
In this repo I'll work with spotify dataset to find out which songs are usually the most popular ones.

Let's imagine situation - a `band / artist / labels` don't know which songs should be `singles / get music video / etc.` There're two solutions: 1 - intuition / experience or 2 - data science.

The question is - Can we estimate a song's popularity from its audio characteristics?
Let's find out by going to notebooks section!

---

All of projects datasets are in gitignore so here's quick guide how to install them! 

1) In this project i'm using spotify dataset taken from kaggle
2) in your terminal write `uv sync`
3) then write `kaggle datasets download solomonameh/spotify-music-dataset`
4) in `data` folder create folder `raw` in which create another folder named `spotify` 
5) go to data unzipper and run it
6) place csv files into `spotify` folder
!!!
7) processed dataset must be loaded from second notebook

---

If you don't use uv or marimo here's the quick guide!
1) use `uv sync` in your terminal, this will install all the packages (if you've already run this before skip this step)
2) go to the notebook 
3) click on the `open as a marimo notebook` in the top right corner OR if there's no button run `uv run python -m marimo edit notebooks\name.py` in terminal.
4) click `Run All` to run all cells and enjoy!