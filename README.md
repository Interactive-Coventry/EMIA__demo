---
emoji: 💻
colorFrom: yellow
colorTo: yellow
sdk: streamlit
sdk_version: 1.22.0
app_file: app.py
pinned: false
---

# EMIA-demo
 Standalone demo for EMIA

- The model is running *LIVE* at [Huggingface](https://huggingface.co/spaces/Interactive-Coventry/EMIA__demo).

- Base repo: [Github | Interactive-Coventry/EMIA__demo](https://github.com/Interactive-Coventry/EMIA__demo/)

  - To run with Streamlit:
    - Run in a terminal with `streamlit run .\streamlit-app.py` and open `http://localhost:8501/` in your browser.


## How to run the demo with Streamlit in a browser with GUI
```
$ streamlit run .\streamlit-app.py
```
![demo_1.png](assets/demo_1.png)
![demo_2.png](assets/demo_2.png)

## How to run from terminal with CLI (command line interface) tool
- Get insights for single image using history <br>
```
    $ python run_emia.py analyze "data\test\1703\1703_20230913183132.jpg" 5
```

- Test insights for single image using history <br>
```
    $ python run_emia.py test 
```

- Get help <br>
```
    $ python run_emia.py --help <br>
    $ python run_emia.py analyze --help
    $ python run_emia.py test --help
```

## For the config.ini file settings 

- device: cpu or cuda
- db_mode: local or streamlit or firebase
- 
