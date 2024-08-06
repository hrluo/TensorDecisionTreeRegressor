
# TensorDecisionTreeRegressor

**Content**
This is the code repository for the research publication "Fast Decision Trees for Tensor Regressions by Hengrui Luo, Akira Horiguchi and Li Ma. 
The manuscript of this paper can be accessed at https://arxiv.org/abs/2408.01926. 

 - We provided a set of illustrative code of tensor decision tree (TT) regressors that serves as a proof of concept, and also a set of robust code that can be executed for large datasets.
 -  Reproducible Code
	 - in Examples_Figure1.ipynb, we provide the code to reproduce coefficient estimation of mixed samples.
	 - in /Example_Figure2, we provide the code and recorded benchmark results for the experiment of comparing different combinations of splitting criteria and optimization methods.
	 - in /Table3, we provide necessary code for comparing tensor-on-tensor regressors on synthetic functions.
	 - in Example_Figure45.ipynb, we provide the code and recorded benchmark results for the experiment of investigating the effect of model ranks on synthetic data.
	 - in /Figure67, we provide the code and recorded benchmark results for the experiment of different depths of the tree, and performance comparison against other methods.
	 - in /Figure8, we provide the script for reproducing the scaling plot of TT models. Note that the results can vary on different machines, this experiments serves as an evidence for complexity analysis in our article. 
 -  General Usage
***scalar-on-tensor regression.***
```
		    model  =  TensorDecisionTreeRegressor(max_depth=3, min_samples_split=2,split_method='variance', split_rank=2, CP_reg_rank=4, Tucker_reg_rank=3, n_mode=3)
			model.use_mean_as_threshold  =  False
			model.sample_rate  =  1
			model.fit(X,y)
			model.print_tree()
			model.prune(X,y,model_type='cp',alpha=0.) 
```

The first line defines a TT model, with max_depth specifying the maximal depth of the generated tree structure; min_sample_split specifying the minimum number of samples needed to perform a split; split_method indicates the splitting criteria and optimization methods. Currently, we support the following options:
		- 'variance', variance splitting criteria with exhaustive search.
		- 'variance_LS', variance splitting criteria with leverage score (LS) sampling.
		- 'lowrank', Low-rank approximation (LAE) splitting criteria with exhaustive search.
		- 'lowrank_reg', Low-rank regression (LRE) splitting criteria with exhaustive search.
		- 'lowrank_LS', LAE splitting criteria with leverage score (LS) sampling.
		- 'lowrank_reg_LS', LRE splitting criteria with leverage score (LS) sampling.
		- 'lowrank_BB', LAE splitting criteria with branch-and-bound (BB).
		- 'lowrank_reg_BB', LAE splitting criteria with branch-and-bound (BB).
Then we also need to specify the associated rank parameters:
		- split_rank: the rank parameter used for computing the splitting criteria like LAE, LRE.
		- CP_reg_rank: the rank parameter in each leaf CP model for the tree.
		- Tucker_reg_rank: the rank parameter in each leaf Tucker model for the tree.
and the n_mode indicating the number of dimensions of the input tensor $X$.
The second line indicates whether we should search for optimal cut points or just use mean threshold corresponding to $\overline{SSE},\overline{LAE},\overline{LRE}$ in our article.
The third line sets parameter sample_rate which only works with the leverage score (LS) sampling optimization.
The fourth line fits the model with given training data $(X,y)$.
The fifth line prints the fitted model.
The sixth line recursively prunes the model with given training data $(X,y)$ and a chosen model_type (corresponding to different kinds of penalty in the complexity measure) and $\alpha$ parameter (also in complexity measure).



***tensor-on-tensor regression.***
```
gradient_boosting_regressor  = GradientBoostingRegressor(n_estimators=10,learning_rate=0.1,weak_learner=model)
gradient_boosting_regressor.pruning =  True

gradient_boosting_regressor.fit(X, y, X_test, y_test)
entrywise_CP  =  gradient_boosting_regressor.predict(X_test,regression_method='cp')
entrywise_Tucker  =  gradient_boosting_regressor.predict(X_test,regression_method='tucker')
```
The GradientBoostingRegressor class accepts the n_estimators as the number of the weak learners we want to put into the ensemble; learning_rate for the update rate and a template weak_learner (e.g., TT model we fitted above). Note that this weak_learner template must be fitted (e.g., via fit method) before.
Then we can set the pruning option to tell the Gradient Boosting to prune each weak learner or not, if the weak_learner template supports the prune method. All weak learners will have the same parameters as the template, like split_rank, n_mode etc.
Using the fit functionality and predict on a new dataset. The training set $(X_{test}, y_{test})$ is optional in the fit method, if supplied, the function will print out-of-sample MSE for reference. 
For more advanced usage of TT for tensor-on-tensor regressions, please refer to tensorOutput_TT_syntheticCP.ipynb and tensorOutput_TT_syntheticTucker.ipynb in the \Table3 folder.


**Abstract**
We proposed the tensor-input tree (TT) method for scalar-on-tensor and tensor-on-tensor regression problems. We first address scalar-on-tensor problem by proposing scalar-output regression tree models whose input variable are tensors (i.e., multi-way arrays). We devised and implemented fast randomized and deterministic algorithms for efficient fitting of scalar-on-tensor trees, making TT competitive against tensor-input GP models. Based on scalar-on-tensor tree models, we extend our method to tensor-on-tensor problems using additive tree ensemble approaches.  Theoretical justification and extensive experiments on real and synthetic datasets are provided to illustrate the performance of TT.

**Citation**
We provided both iPynb illustrative code, Python production code for reproducible and experimental purposes under [LICENSE](https://github.com/hrluo/TensorDecisionTreeRegressor/blob/master/LICENSE).
Please cite our paper using following BibTeX item:

    @article{luo2024tensortree,
		title={Efficient Decision Trees for Tensor Regressions}, 
		author={Hengrui Luo and Akira Horiguchi and Li Ma},
		year={2024},
		eprint={2408.01926},
		archivePrefix={arXiv},
		primaryClass={cs.LG},
		url={https://arxiv.org/abs/2408.01926}, 
    }

Thank you again for the interest and please reach out if you have further questions.
