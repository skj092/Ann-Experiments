	?#d ???#d ??!?#d ??	?g??[`@?g??[`@!?g??[`@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?#d ??<?b??*??A?E????Y??!o????rEagerKernelExecute 0*	??Q?V@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat ?????!C?s9??A@)??????1?????>@:Preprocessing2U
Iterator::Model::ParallelMapV2^?/?ۖ?!$n N9@)^?/?ۖ?1$n N9@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???ܑ?!yi???3@)d?w???1/Lθ?G*@:Preprocessing2F
Iterator::ModelU4??????!j?%?B@)?"?J %??1????(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?٭e2??!??]??7O@)?'?_{?1`???N@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??(]?w?!??p?@)??(]?w?1??p?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?PۆQp?!{?'@)?PۆQp?1{?'@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s9.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?g??[`@I?)D??W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	<?b??*??<?b??*??!<?b??*??      ??!       "      ??!       *      ??!       2	?E?????E????!?E????:      ??!       B      ??!       J	??!o??????!o????!??!o????R      ??!       Z	??!o??????!o????!??!o????b      ??!       JCPU_ONLYY?g??[`@b q?)D??W@