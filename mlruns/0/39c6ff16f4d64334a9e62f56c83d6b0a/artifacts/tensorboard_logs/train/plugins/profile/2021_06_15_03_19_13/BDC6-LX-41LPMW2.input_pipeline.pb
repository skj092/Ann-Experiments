	KVE?Ɉ@KVE?Ɉ@!KVE?Ɉ@	몣F?@몣F?@!몣F?@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:KVE?Ɉ@?t???l??A??8ӄ-@YBv??fG??rEagerKernelExecute 0*	?O??n>i@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??r?m???!??6|	B@)ѱ?J\ǰ?1?aZ*:@@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapA?ȓ?k??![??k=@)?n/i?֩?1??պ?8@:Preprocessing2U
Iterator::Model::ParallelMapV2Nd??Ǣ?!v?:?)2@)Nd??Ǣ?1v?:?)2@:Preprocessing2F
Iterator::Model????U??!??J.?o:@)?9???146?毌 @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?;???!?XmtdR@)??{?_???1??wg?" @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??d?`T??!???j!?@)??d?`T??1???j!?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorS??.?}?!?????@)S??.?}?1?????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9몣F?@IQŕ???W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?t???l???t???l??!?t???l??      ??!       "      ??!       *      ??!       2	??8ӄ-@??8ӄ-@!??8ӄ-@:      ??!       B      ??!       J	Bv??fG??Bv??fG??!Bv??fG??R      ??!       Z	Bv??fG??Bv??fG??!Bv??fG??b      ??!       JCPU_ONLYY몣F?@b qQŕ???W@