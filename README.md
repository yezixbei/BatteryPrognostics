#Battery Prognostics via Deep Learning

##Abstract

The purpose of this tool is to use capacity to estimate the remaining useful life of a rechargeable battery. It was built for Professor Charniak at Brown University while I was completing a masters in computer science. The purpose was to extend the niche knowledge I gained working as a process engineer at a battery start up prior to grad school. 



##Introduction

The state of the art lithium cobalt batteries used in consumer electronics are made of a carbon anode, a lithium cobalt oxide cathode, plastic separators, and liquid electrolyte. We have to take into account these material properties and the interactions amongst them along their interfaces. Instead of creating a complex electrochemical model of the system in order to predict its behavior, which would be time consuming, expensive, and inaccurate, a neural net is an appropriate tool for this purpose. Using neural nets, we can treat the data source as a black box and predict its behavior using preexisting data, which should exist in droves due to the proliferation of data intensive work on battery cycling from companies such as Tesla and Apple.



##Experimental Procedures

###Data Source & Concerns related to Battery Degradation

The data for this experiment comes from performing an aging test on standard lithium-ion cells, which may come in multiple sizes such as button, coin, or cylinder, and in different material configurations. Because many factors play a role in the decay of battery performance, we need equipment to generate testing cells and to control ambient and charging condition. We need a glove box, fume hoods, a battery cycler, and a temperature-controlled chamber in order to generate meaningful experimental data, which requires high capital investment.  My work is based on a paper written in 2017 where the authors of the paper generated their data at the Center of Innovation for Electric Vehicles in Beijing(1). Since they did not make their raw data available, I found a dataset from the NASA Prognostic Center that closely matched their experimental conditions, instead of generating the data myself, which would have been impossible.(2)

Like in the original study, the dataset from NASA was generated using 18650 cylindrical lithium ion cells cycled about 600 times, from 2.5 to 4.2V, at 25C, and at a comparable charging rate of 3/4C and a discharging rate of 1C. The original study used LiCoO2 alloyed with Ni and Al, which is a well-studied, industry standard practice to reduce material resistance. Although the Li-ion electrode compositions were not reported in the NASA study, we can assume they are comparable based on their cycling limit. The upper and lower cycling limits depends on battery material; for example, LiCoO2 is stable at an upper limit of 4.2V, while mixtures of LiCoO2 and say, Mn, can be cycled at as high as 4.5V. Expanding the charge and discharge limit of a particular material will drastically degrade the material structure, leading to catastrophic failure.  Cycling rate and cycling temperature also influence battery degradation and need to be controlled. Higher temperature will lead to faster degradation. Pre-cycled electrodes are thermal dynamically less stable than their cycled counterparts; increasing ambient temperature will lower the activation energy of many undesirable processes that degrade battery life, such as the break down of the electrode and electrolyte interface. Due to all these issues, I was careful to find a dataset that closely replicated the original experimental conditions. 

###Data Cleaning & NN Architecture

Capacity is defined as the capacity of a battery to store charge. It can be a proxy for battery health, since its degradation over time can be replicated by holding cycling and ambient conditions constant. Failure is predefined to be about 0.8 normalized capacity. Capacity is measured in Ah, and is the area under a charging or discharging curve. To extracted capacity data over time, I started with the charging curves from the corresponding JSON file, removed any outliers that are over two standard deviation from the mean over an linear interval of ten data points, and integrated the charging curve over number of cycles and normalized it against the first cycle in order to see the capacity degradation. In Fig. 1, You can see that the normalize capacity degrades from 1 to 0.75 over a span of about 600 cycles. The purpose of the tool is to predict the exact cycle at which normalized capacity will hit 0.8. 

My neural net consists of a single GRU layer with 128 neurons, followed by one fully connected normal layer. The original paper uses two LSTM Layers, but I liked GRUs better. However, I did use the same train-test ratio and the same optimizer. I divided a single capacity curve, which corresponds to the decay of a single cell, into train and test portions, then in the training portion, I subdivided it again into n non-overlapping windows, rolled them n times, and then unrolled them to predict the (n+1)st window. For more information about how this works, please refer to the reference section.



##Results & Conclusions

In Fig. 2  you can see that the difference between the predicted cycle at which the battery fails and the actual cycle is about 8 over a life time of 600. You can experiment with the tool using the JSON files included for the two different cells(B0006 and B0007) or using any other capacity data. In lieu of capacity degradation, charge transfer resistance can also be used as a proxy. (CTR increases over time.) The code itself is not complicated enough to be broken into multiple modules; however, I may fix that in a future commit in order to follow good software engineering practices.  One possible extension of this into a fully-fledged tool is to add an alert feature that looks ahead, given real time charging curves of a single battery, which should be useful in an electric car or a consumer electronic device. 



##References

1. Y Zhang, R Xiong, H He, Z Liu (2017) “A LSTM-RNN method for the lithium ion battery remaining useful life prediction” Prognostics and System Health Management Conference (Harbin)
2. B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data Repository , NASA Ames Research Center, Moffett Field, CA
3. L. Weng (2017) “ Predict Stock Prices Using RNN”
(https://lilianweng.github.io/lil-log/2017/07/08/predict-stock-prices-using-RNN-part-1)



## License

This project is licensed under the MIT License


