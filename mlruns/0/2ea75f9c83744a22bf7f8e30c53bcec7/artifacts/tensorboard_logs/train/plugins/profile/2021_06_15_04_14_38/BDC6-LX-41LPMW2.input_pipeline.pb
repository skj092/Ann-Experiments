	???1v??????1v???!???1v???	R??n?p@R??n?p@!R??n?p@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:???1v???????	??AFCƣT???Y??????rEagerKernelExecute 0*	&1?h@2U
Iterator::Model::ParallelMapV2?}??g??!?q);?<@)?}??g??1?q);?<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?R\U?]??!?U??9@)?ݭ,?Y??1?G????6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?k$	???!|8???e8@)??6?????1y?4??2@:Preprocessing2F
Iterator::Model????V`??!!?y??D@)?V??,???1F?ԓ)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip/m8,???!??Q?\M@)7?????1W_?:A!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?E???Ԉ?!hj?L'@)?E???Ԉ?1hj?L'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??sE)!x?!q??#q@)??sE)!x?1q??#q@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t10.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9R??n?p@I????(W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????	??????	??!????	??      ??!       "      ??!       *      ??!       2	FCƣT???FCƣT???!FCƣT???:      ??!       B      ??!       J	????????????!??????R      ??!       Z	????????????!??????b      ??!       JCPU_ONLYYR??n?p@b q????(W@