To run this application on Ubuntu (WSL works):

WSL Ubuntu is missing some basic software:

`apt install python3-dev python3-venv`

Clone the repository and cd into it. Create a new venv environment called venv:

`python3 -m venv venv`

And activate it:

`source venv/bin/activate`

Then install the required packages:

`pip install -r requirements.txt`

And run streamlit!

`streamlit run Overview_and_Clustering.py`
