	?d?????d????!?d????	?~?w)@?~?w)@!?~?w)@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?d?????q6??A?+?,???Y}??O9&??rEagerKernelExecute 0*	a??"?IR@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?º??Ș?!??i?@@)ʋL?????1???au;@:Preprocessing2U
Iterator::Model::ParallelMapV2??>????!???Y?X:@)??>????1???Y?X:@:Preprocessing2F
Iterator::ModelF?~ໝ?!???s??C@)ѯ?????1??V?*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap&n?@׎?!6?>?ؕ4@)? l@????1ZeO???(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??Ry=x?!y.?- @)??Ry=x?1y.?- @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip0e?????!`?L'N@)c????s?1?????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorUܸ???p?!?D???@)Uܸ???p?1?D???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t16.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?~?w)@IH?Hn}W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?q6???q6??!?q6??      ??!       "      ??!       *      ??!       2	?+?,????+?,???!?+?,???:      ??!       B      ??!       J	}??O9&??}??O9&??!}??O9&??R      ??!       Z	}??O9&??}??O9&??!}??O9&??b      ??!       JCPU_ONLYY?~?w)@b qH?Hn}W@