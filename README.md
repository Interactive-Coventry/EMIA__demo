---
emoji: 💻
colorFrom: yellow
colorTo: yellow
sdk: gradio
sdk_version: 3.47.1
app_file: app.py
pinned: false
---

# EMIA-demo
 Standalone demo for EMIA

- The model is running *LIVE* at [Huggingface](https://huggingface.co/spaces/Interactive-Coventry/EMIA__demo).

- Base repo: [Github | Interactive-Coventry/EMIA__demo](https://github.com/Interactive-Coventry/EMIA__demo/)

  - Run in a terminal with `python app.py` and open `http://localhost:7860` in your browser.
  
  - Alternatively, run with `gradio app.py` to start a dev server with hot reloading enabled.
    
![demo_1.png](assets/demo_1.png)


# How to run the demo with Gradio in a browser with GUI
```
$python app.py
$gradio app.py
```

# How to run from terminal with CLI (command line interface) tool
- Get insights for single image using history <br>
```
    $python run_emia.py analyze "data\test\1703\1703_20230913183132.jpg" 5
```

- Test insights for single image using history <br>
```
    $python run_emia.py test 
```

- Get help <br>
```
    $python run_emia.py --help <br>
    $python run_emia.py analyze --help
    $python run_emia.py test --help
```
