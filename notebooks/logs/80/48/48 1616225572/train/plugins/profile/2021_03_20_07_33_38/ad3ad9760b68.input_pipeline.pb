	z�):���?z�):���?!z�):���?	X�	�"I@X�	�"I@!X�	�"I@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$z�):���?�(A�G�?AL����?Yw.�����?*	ףp=�e@2U
Iterator::Model::ParallelMapV2l���D�?!����:@)l���D�?1����:@:Preprocessing2F
Iterator::ModeleM�?!O!���8I@)��j�?1�)��\8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat1'h��'�?!$��7@){K9_콠?16L�=�2@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��S:X�?!^��o��!@)��S:X�?1^��o��!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateJ&�v���?!��"A�0@)�g�����?1/»$�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��(��?!��)0�H@)DN_��,�?1z���5@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�n��\��?!��A�@)�n��\��?1��A�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�jQLޠ?!��=D�3@)4��HLp?1�?��n@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 50.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2t11.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9W�	�"I@I��W�	�H@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�(A�G�?�(A�G�?!�(A�G�?      ��!       "      ��!       *      ��!       2	L����?L����?!L����?:      ��!       B      ��!       J	w.�����?w.�����?!w.�����?R      ��!       Z	w.�����?w.�����?!w.�����?b      ��!       JCPU_ONLYYW�	�"I@b q��W�	�H@