"?=
uHostFlushSummaryWriter"FlushSummaryWriter(1??~j???@9??~j???@A??~j???@I??~j???@a֒G?Ю??i֒G?Ю???Unknown?
BHostIDLE"IDLE1?p=
??@A?p=
??@aQ????+??i??yD?m???Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1}?5^??r@9}?5^??r@A}?5^??r@I}?5^??r@aH7?????iuH????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1R????p@9R????p@AR????p@IR????p@a?f???]??ix[C??????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1D?l??Qe@9D?l??Qe@AD?l??Qe@ID?l??Qe@a?	XRc???i??̺????Unknown
^HostGatherV2"GatherV2(1^?I+a@9^?I+a@A^?I+a@I^?I+a@a??q[? ??iƩ??6???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1?rh???S@9?rh???S@A?rh???S@I?rh???S@a???O??i-?K^֎???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1+??D@9+??D@A+??D@I+??D@a?h??G%w?i??x? ????Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1o??ʡD@9o??ʡD@Ao??ʡD@Io??ʡD@a??!?\?v?i$;??????Unknown
i
HostWriteSummary"WriteSummary(1h??|??C@9h??|??C@Ah??|??C@Ih??|??C@ava?>?u?i?Z?$????Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1V-?C@9V-?C@A?z?GQA@I?z?GQA@a??B'+s?ik?s?<???Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1?v??O>@9?v??O>@A?v??O>@I?v??O>@aJĜ?<?p?i?Ǥ?t^???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?x?&1?B@9?x?&1?B@A???(\?<@I???(\?<@a?????p?iq??|~???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1`??"?y;@9`??"?y;@A`??"?y;@I`??"?y;@a?J?w?in?i????????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1?"??~J6@9?"??~J6@A?"??~J6@I?"??~J6@a??v?h?iͼ?9?????Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1P??n?4@9P??n?4@AP??n?4@IP??n?4@a?L1?[?f?iF????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1????M1@9????M1@A????M1@I????M1@a?ݹP??b?i?? R????Unknown
dHostDataset"Iterator::Model(1???QX@@9???QX@@A?&1?\/@I?&1?\/@a?M$?m[a?iF̭?u????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1?z?G?+@9?z?G?+@A?z?G?+@I?z?G?+@ax?/?^?iM9??????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1???K7V@9???K7V@A?K7?A`&@I?K7?A`&@a?7^???X?i0|?F???Unknown
gHostTanh"sequential/dense/Tanh(1?E???T&@9?E???T&@A?E???T&@I?E???T&@a?????X?iu??'????Unknown
`HostGatherV2"
GatherV2_1(1R???Q&@9R???Q&@AR???Q&@IR???Q&@a???T??X?iQ6?$???Unknown
aHostIdentity"Identity(1P??nC&@9P??nC&@AP??nC&@IP??nC&@aj??>?X?i???N1???Unknown?
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1????K?%@9????K?%@A????K?%@I????K?%@a?
?z?	X?iaTbS=???Unknown
eHost
LogicalAnd"
LogicalAnd(1u?V?$@9u?V?$@Au?V?$@Iu?V?$@aal?	?W?i?L?7?H???Unknown?
YHostPow"Adam/Pow(1q=
ף?!@9q=
ף?!@Aq=
ף?!@Iq=
ף?!@af????S?i?????R???Unknown
ZHostArgMax"ArgMax(1??Q?E!@9??Q?E!@A??Q?E!@I??Q?E!@a/?ʍ?S?i???S\???Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1????K7!@9????K7!@A????K7!@I????K7!@a?UdS?i???e???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(15^?I? @95^?I? @A5^?I? @I5^?I? @aC??3?ER?i??s??n???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1???Q?@9???Q?@A???Q?@I???Q?@a?8??&?Q?i?>???w???Unknown
[HostAddV2"Adam/add(1NbX94@9NbX94@ANbX94@INbX94@a%??
EQ?iG?/?g????Unknown
[ HostPow"
Adam/Pow_1(1B`??"[@9B`??"[@AB`??"[@IB`??"[@a+χ?GN?i?q?????Unknown
?!HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1???K7	@9???K7	@A???K7	@I???K7	@aۺ|8??L?i;?_?-????Unknown
t"HostAssignAddVariableOp"AssignAddVariableOp(1??? ??@9??? ??@A??? ??@I??? ??@a?_????L?i?>?	\????Unknown
V#HostSum"Sum_2(1      @9      @A      @I      @a???s??J?i@)?8 ????Unknown
l$HostIteratorGetNext"IteratorGetNext(1j?t?@9j?t?@Aj?t?@Ij?t?@a[_??yI?ia?^????Unknown
?%HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1'1??@9'1??@A'1??@I'1??@a??Y??G?i??\?#????Unknown
o&HostReadVariableOp"Adam/ReadVariableOp(1=
ףp=@9=
ףp=@A=
ףp=@I=
ףp=@a$D?HgF?i?????????Unknown
?'HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?I+@9?I+@A?I+@I?I+@a??N6+F?iH|H????Unknown
?(HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1????M?@9????M?@A????M?@I????M?@a???hF?i?!ɹ???Unknown
~)HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1B`??"[@9B`??"[@AB`??"[@IB`??"[@a??@a?lE?i?q?K$????Unknown
}*HostTanhGrad"'gradient_tape/sequential/dense/TanhGrad(1??S??@9??S??@A??S??@I??S??@a??y???B?i&P&>?????Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_4(1T㥛? @9T㥛? @AT㥛? @IT㥛? @a)?O??A?i ?
?V????Unknown
?,HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1?~j?t?@9?~j?t?@A?~j?t?@I?~j?t?@a?g??@?i?Kȑ????Unknown
v-HostCast"$sparse_categorical_crossentropy/Cast(1??ʡE?@9??ʡE?@A??ʡE?@I??ʡE?@aH?OR????i?]uƊ????Unknown
?.HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1?S㥛?
@9?S㥛?
@A?S㥛?
@I?S㥛?
@aՂV?=?i. ?>????Unknown
X/HostEqual"Equal(1??ʡE
@9??ʡE
@A??ʡE
@I??ʡE
@a;??=?i??=z?????Unknown
`0HostDivNoNan"
div_no_nan(1??n??@9??n??@A??n??@I??n??@aW?Xd?e;?i???4N????Unknown
]1HostCast"Adam/Cast_1(1o??ʡ@9o??ʡ@Ao??ʡ@Io??ʡ@a??*9?iA?o?o????Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_3(1?? ?rh@9?? ?rh@A?? ?rh@I?? ?rh@a?k???8?i?F.??????Unknown
t3HostReadVariableOp"Adam/Cast/ReadVariableOp(1??n??@9??n??@A??n??@I??n??@a?????8?i??????Unknown
?4HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1+????@9+????@A+????@I+????@a??(??57?i#???r????Unknown
X5HostCast"Cast_2(1+??@9+??@A+??@I+??@aOL8đ7?i-'U????Unknown
b6HostDivNoNan"div_no_nan_1(1L7?A`?@9L7?A`?@AL7?A`?@IL7?A`?@aL(tq?4?i.?Ju?????Unknown
T7HostMul"Mul(1`??"??@9`??"??@A`??"??@I`??"??@a2c????3?i??=-o????Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_1(1???S????9???S????A???S????I???S????a?U??1?i?????????Unknown
v9HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1}?5^?I??9}?5^?I??A}?5^?I??I}?5^?I??a??e??O/?i?]??????Unknown
X:HostCast"Cast_3(1?ʡE????9?ʡE????A?ʡE????I?ʡE????a?^??.?i?????????Unknown
?;HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1P??n???9P??n???AP??n???IP??n???akx??!+?i{???5????Unknown
y<HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1?E??????9?E??????A?E??????I?E??????a?mUe?E)?i?6?*?????Unknown
?=HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1y?&1???9y?&1???Ay?&1???Iy?&1???a?$?l)?i$8ʱ[????Unknown
w>HostReadVariableOp"div_no_nan/ReadVariableOp_1(1?~j?t???9?~j?t???A?~j?t???I?~j?t???a????8?(?iX??????Unknown
w?HostReadVariableOp"div_no_nan_1/ReadVariableOp(1??? ?r??9??? ?r??A??? ?r??I??? ?r??a-%????(?i???y????Unknown
v@HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1R???Q??9R???Q??AR???Q??IR???Q??a?(?
?}&?iA???????Unknown
uAHostReadVariableOp"div_no_nan/ReadVariableOp(1h??|?5??9h??|?5??Ah??|?5??Ih??|?5??a?
????!?i     ???Unknown*?<
uHostFlushSummaryWriter"FlushSummaryWriter(1??~j???@9??~j???@A??~j???@I??~j???@aR~D?X???iR~D?X????Unknown?
rHost_FusedMatMul"sequential/dense/BiasAdd(1}?5^??r@9}?5^??r@A}?5^??r@I}?5^??r@a??'#5??i?y?<?????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1R????p@9R????p@AR????p@IR????p@a??Ѱ%??it???A????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1D?l??Qe@9D?l??Qe@AD?l??Qe@ID?l??Qe@a3??????i7??c????Unknown
^HostGatherV2"GatherV2(1^?I+a@9^?I+a@A^?I+a@I^?I+a@a??	?@??iG?e?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1?rh???S@9?rh???S@A?rh???S@I?rh???S@a8t?ؐ?i?2?M|C???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1+??D@9+??D@A+??D@I+??D@a??r?鴁?i,???O????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1o??ʡD@9o??ʡD@Ao??ʡD@Io??ʡD@a~2cd?x??i??_2????Unknown
i	HostWriteSummary"WriteSummary(1h??|??C@9h??|??C@Ah??|??C@Ih??|??C@a?F?ɽ??i9??D)???Unknown?
?
HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1V-?C@9V-?C@A?z?GQA@I?z?GQA@a#?<7T}?iWg?M???Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1?v??O>@9?v??O>@A?v??O>@I?v??O>@a???[J?y?i????%????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?x?&1?B@9?x?&1?B@A???(\?<@I???(\?<@a՜\dp?x?i8???&????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1`??"?y;@9`??"?y;@A`??"?y;@I`??"?y;@az2\)Dw?iq</?????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1?"??~J6@9?"??~J6@A?"??~J6@I?"??~J6@a??zf,?r?i? 	?o???Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1P??n?4@9P??n?4@AP??n?4@IP??n?4@a?١?^q?i]D?,)???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1????M1@9????M1@A????M1@I????M1@a???P?l?i? ???E???Unknown
dHostDataset"Iterator::Model(1???QX@@9???QX@@A?&1?\/@I?&1?\/@a?=x??j?i-du?`???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1?z?G?+@9?z?G?+@A?z?G?+@I?z?G?+@a9?J??g?iI??2%x???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1???K7V@9???K7V@A?K7?A`&@I?K7?A`&@a]?w??b?i.?%?????Unknown
gHostTanh"sequential/dense/Tanh(1?E???T&@9?E???T&@A?E???T&@I?E???T&@a?????b?i??? ????Unknown
`HostGatherV2"
GatherV2_1(1R???Q&@9R???Q&@AR???Q&@IR???Q&@a?fz?u?b?i?9?Q?????Unknown
aHostIdentity"Identity(1P??nC&@9P??nC&@AP??nC&@IP??nC&@a(?z??b?iV??4?????Unknown?
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1????K?%@9????K?%@A????K?%@I????K?%@a?}?׆cb?iԥ{?$????Unknown
eHost
LogicalAnd"
LogicalAnd(1u?V?$@9u?V?$@Au?V?$@Iu?V?$@a?a?B?a?i????????Unknown?
YHostPow"Adam/Pow(1q=
ף?!@9q=
ף?!@Aq=
ף?!@Iq=
ף?!@a?W?a^?i
?'??????Unknown
ZHostArgMax"ArgMax(1??Q?E!@9??Q?E!@A??Q?E!@I??Q?E!@a?Ӹ3y?]?it???????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1????K7!@9????K7!@A????K7!@I????K7!@a?>?(]?ivl??'???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(15^?I? @95^?I? @A5^?I? @I5^?I? @a??? ?[?i??"J""???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1???Q?@9???Q?@A???Q?@I???Q?@a?0??)?Z?i?F?^?/???Unknown
[HostAddV2"Adam/add(1NbX94@9NbX94@ANbX94@INbX94@a3 GNlZ?i????<???Unknown
[HostPow"
Adam/Pow_1(1B`??"[@9B`??"[@AB`??"[@IB`??"[@a??p?%*W?i??[H???Unknown
? HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1???K7	@9???K7	@A???K7	@I???K7	@a??!? V?i?r?aS???Unknown
t!HostAssignAddVariableOp"AssignAddVariableOp(1??? ??@9??? ??@A??? ??@I??? ??@a'	r.??U?i Y	^^???Unknown
V"HostSum"Sum_2(1      @9      @A      @I      @a??'?RT?i?ߜb?h???Unknown
l#HostIteratorGetNext"IteratorGetNext(1j?t?@9j?t?@Aj?t?@Ij?t?@a?hu0V}S?iT?Fr???Unknown
?$HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1'1??@9'1??@A'1??@I'1??@a??D;?Q?i?'W+{???Unknown
o%HostReadVariableOp"Adam/ReadVariableOp(1=
ףp=@9=
ףp=@A=
ףp=@I=
ףp=@a??f?#Q?i?.???????Unknown
?&HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?I+@9?I+@A?I+@I?I+@a?I?{??P?i?H?&????Unknown
?'HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1????M?@9????M?@A????M?@I????M?@aPߋ8d?P?i?ad??????Unknown
~(HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1B`??"[@9B`??"[@AB`??"[@IB`??"[@aOYl??cP?i????Ü???Unknown
})HostTanhGrad"'gradient_tape/sequential/dense/TanhGrad(1??S??@9??S??@A??S??@I??S??@aOdH?<?L?i?:????Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_4(1T㥛? @9T㥛? @AT㥛? @IT㥛? @a???ZPK?iQ??P֪???Unknown
?+HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1?~j?t?@9?~j?t?@A?~j?t?@I?~j?t?@a?8
~+?I?i?S?[O????Unknown
v,HostCast"$sparse_categorical_crossentropy/Cast(1??ʡE?@9??ʡE?@A??ʡE?@I??ʡE?@ar-?PH?i?x?ac????Unknown
?-HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1?S㥛?
@9?S㥛?
@A?S㥛?
@I?S㥛?
@a??????F?i?t?????Unknown
X.HostEqual"Equal(1??ʡE
@9??ʡE
@A??ʡE
@I??ʡE
@ayx??)?F?il?؝????Unknown
`/HostDivNoNan"
div_no_nan(1??n??@9??n??@A??n??@I??n??@ae??M??D?iW`B?????Unknown
]0HostCast"Adam/Cast_1(1o??ʡ@9o??ʡ@Ao??ʡ@Io??ʡ@a M?'*C?i/@jȥ????Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_3(1?? ?rh@9?? ?rh@A?? ?rh@I?? ?rh@aUR*M??B?iĊ?*d????Unknown
t2HostReadVariableOp"Adam/Cast/ReadVariableOp(1??n??@9??n??@A??n??@I??n??@aqŨTkB?i
????????Unknown
?3HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1+????@9+????@A+????@I+????@a?*?`?A?i??Xo????Unknown
X4HostCast"Cast_2(1+??@9+??@A+??@I+??@a
?XI]?A?i?e/?????Unknown
b5HostDivNoNan"div_no_nan_1(1L7?A`?@9L7?A`?@AL7?A`?@IL7?A`?@a??1G7 @?iA?6=?????Unknown
T6HostMul"Mul(1`??"??@9`??"??@A`??"??@I`??"??@a??y??q>?ix6o?????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_1(1???S????9???S????A???S????I???S????au?Q??:?i?>? ????Unknown
v8HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1}?5^?I??9}?5^?I??A}?5^?I??I}?5^?I??a?I,4/?7?i@?܆?????Unknown
X9HostCast"Cast_3(1?ʡE????9?ʡE????A?ʡE????I?ʡE????ahQu?X?7?i?R???????Unknown
?:HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1P??n???9P??n???AP??n???IP??n???a$???4?ilWK&?????Unknown
y;HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1?E??????9?E??????A?E??????I?E??????a06?!sU3?i?????????Unknown
?<HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1y?&1???9y?&1???Ay?&1???Iy?&1???aV:??23?i??--^????Unknown
w=HostReadVariableOp"div_no_nan/ReadVariableOp_1(1?~j?t???9?~j?t???A?~j?t???I?~j?t???am?q?3?i????????Unknown
w>HostReadVariableOp"div_no_nan_1/ReadVariableOp(1??? ?r??9??? ?r??A??? ?r??I??? ?r??a??53?i?)_2"????Unknown
v?HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1R???Q??9R???Q??AR???Q??IR???Q??aL9??41?i??^?H????Unknown
u@HostReadVariableOp"div_no_nan/ReadVariableOp(1h??|?5??9h??|?5??Ah??|?5??Ih??|?5??a?2?
s+?i      ???Unknown2CPU