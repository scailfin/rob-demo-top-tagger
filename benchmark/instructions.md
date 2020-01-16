### Benchmark Goals

Based on the established task of identifying boosted, hadronically decaying top quarks, this benchmark compares a wide range of modern machine learning approaches (see [The Machine Learning Landscape of Top Taggers](https://arxiv.org/abs/1902.09914) paper for more details).

The goal of this study is to see how well different neutral network setups can classify jets based on calorimeter information. While initially it was not clear if any of the machine learning methods applied to toptagging would be able to significantly exceed the performance of the multi-variate tools,later studies have consistently showed that we can expect great performance improvement from most modern tools. This turns around the question into which of the tagging approaches have the best performance (also relative to their training effort), and if the leading taggers make use of the same, hence complete set of information.

### How to Participate

The benchmark workflow consists of three main steps.

![Benchmark Workflow Overview](https://github.com/scailfin/rob-demo-top-tagger/raw/master/docs/graphics/toptagger-overview.png "Top Tagger Benchmark - Workflow Overview")

Participants are given a test dataset consisting of 200k signal and 200k background jets. The top signal and mixed quark-gluon background jets are produced with using Pythia8 with its default tune for a center-of-mass energy of 14 TeV and ignoring multiple interactions and pile-up. For a simplified detector simulation we use Delphes with the default ATLAS detector card.

The produced results should contain classification results for each jet to measure the performance of the network and test which jets are correctly classified in each approach. Overall results are sorted in decreasing order by the background rejection as signal efficiency at 50%.
