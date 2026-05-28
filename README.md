# song-popularity
In this repo I'll work with spotify dataset to find out which songs are usually the most popular ones.

Let's imagine situation - a band/artist want to create a popular song, but they don't know is this supposed to be loud or calm, rock or pop etc. This moment I'll come and will have answer for their problem.

So let's answer this question together by going to notebooks section! (not finished project)

---

All of projects datasets are in gitignore so here's guide how to install them! 

1) In this project i'm using spotify dataset taken from kaggle
2) in your terminal write `uv sync`
3) then write `kaggle datasets download solomonameh/spotify-music-dataset`
4) in `data` folder create folder `raw` in which create another folder named `spotify` 
5) go to data unzipper and run it
6) place csv files into `spotify` folder

---

If you don't use uv or marimo here's the quick guide!
1) use `uv sync` in your terminal, this will install all the packages (if you've already run this before skip this step)
2) go to the notebook 
3) click on the `open as a marimo notebook` in the top right corner OR if there's no button run `uv run python -m marimo edit notebooks\name.py` in terminal.
4) click `Run All` to run all cells and enjoy!

On 3 step you might get an error, if so try: `marimo convert .\notebooks\old_name.py -o new_name.py`
This will create new file which will be already marimo notebook