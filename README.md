# TeacherStudentBuildingExtraction

This repo provides the sample code to collect data and find 65 lightweight CNNs used as encoders for 8 encoder-decoder networks for the following paper that is currently "submitted" to an academic journal for possible publication.
"A Comparative Study of Knowledge Transfer Methods for Misaligned Urban Building Labels". The paper is archived at [https://arxiv.org/abs/2303.09064](https://arxiv.org/abs/2311.03867)

Please navigate to NearMapDataCollector folder for the source codes to collect the dataset. For the models, please go to TeacherStudents folder. The codes for domain adaptation, knowledge distillation, and deep mutual learning will be available in this repo after the paper is published.

If you use the codes from the repo, we appreciate your citation to the paper as:

```json
@misc{neupane2023comparative,
      title={A Comparative Study of Knowledge Transfer Methods for Misaligned Urban Building Labels}, 
      author={Bipul Neupane and Jagannath Aryal and Abbas Rajabifard},
      year={2023},
      eprint={2311.03867},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
Acknowledgement:
The authors would like to thank Nearmap for providing the API service to collect the image data for the experiments. Further, the authors also express their special gratitude to Segmentation Models Pytorch and Hugging Face for their continuous support towards open and accessible AI - the extensive study would not have been possible without their commendable work.
