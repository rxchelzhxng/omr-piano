#omr-piano

Optical Music Recognition (OMR) web application that transcribes musical notes on monophonic scores to ABC notation and annotates them onto the scores for aiding the process of learning piano.

![Annotated example](https://github.com/rxchelzhxng/omr-piano/blob/main/examples/annotated.jpg)

# How it works
The web app is created with Flask on a tensorflow model built by Calvo-Zaragoza et al. on their work [End-to-End Neural Optical Music Recognition of Monophonic Scores](https://www.mdpi.com/2076-3417/8/4/606).

# Running the app
1. Download the [semantic model](https://github.com/OMR-Research/tf-end-to-end) and place them in a folder named `semantic/` from the app root directory
2. Install the dependencies: flask, Pillow, opencv, tensorflow
3. Run the command `python app.py` in the directory with the `app.py` file and follow the link to view the web app
4. Upload your music sheet as an image and the annotated sheet will be displayed in the **Result** section and saved with the name `annotated.png`

