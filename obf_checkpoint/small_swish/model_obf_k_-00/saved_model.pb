µ	
í
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.02v2.7.0-rc1-69-gc256c071bb28´Ò
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	(*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@(*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@(*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:(*
dtype0

NoOpNoOp
§%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*â$
valueĜ$BĠ$ BÎ$
Ú
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
R
%	variables
&trainable_variables
'regularization_losses
(	keras_api
R
)	variables
*trainable_variables
+regularization_losses
,	keras_api
h

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
R
3	variables
4trainable_variables
5regularization_losses
6	keras_api
R
7	variables
8trainable_variables
9regularization_losses
:	keras_api
h

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
R
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
8
0
1
2
 3
-4
.5
;6
<7
8
0
1
2
 3
-4
.5
;6
<7
 
­
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
­
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
­
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
­
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
!	variables
"trainable_variables
#regularization_losses
 
 
 
­
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
%	variables
&trainable_variables
'regularization_losses
 
 
 
­
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
)	variables
*trainable_variables
+regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
­
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
/	variables
0trainable_variables
1regularization_losses
 
 
 
­
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
3	variables
4trainable_variables
5regularization_losses
 
 
 
­
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
7	variables
8trainable_variables
9regularization_losses
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1

;0
<1
 
­
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
=	variables
>trainable_variables
?regularization_losses
 
 
 
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
 
N
0
1
2
3
4
5
6
7
	8

9
10
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
~
serving_default_dense_inputPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(*
dtype0*
shape:˙˙˙˙˙˙˙˙˙(
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_inputdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_4354552
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ğ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_4355196

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_4355230ä
Ò	
ĝ
D__inference_dense_1_layer_call_and_return_conditional_losses_4354138

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¤
G
+__inference_dropout_2_layer_call_fn_4354874

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_4354196`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙@:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Û
b
D__inference_dropout_layer_call_and_return_conditional_losses_4354762

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ç	
ġ
D__inference_dense_3_layer_call_and_return_conditional_losses_4354915

inputs0
matmul_readvariableop_resource:@(-
biasadd_readvariableop_resource:(
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs



$__inference_internal_grad_fn_4354974
result_grads_0
result_grads_1&
"sigmoid_sequential_dense_1_biasadd
identityz
SigmoidSigmoid"sigmoid_sequential_dense_1_biasadd^result_grads_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
mulMul"sigmoid_sequential_dense_1_biasaddsub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
addAddV2add/x:output:0mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙U
mul_1MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_2Mulresult_grads_0	mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
IdentityIdentity	mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*O
_input_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:X T
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_1:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Û
b
D__inference_dropout_layer_call_and_return_conditional_losses_4354126

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ë	
ö
D__inference_dense_2_layer_call_and_return_conditional_losses_4354854

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Î	
p
$__inference_internal_grad_fn_4355034
result_grads_0
result_grads_1
sigmoid_inputs
identitye
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@U
mulMulsigmoid_inputssub:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?W
addAddV2add/x:output:0mul:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@T
mul_1MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Y
mul_2Mulresult_grads_0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Q
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
identityIdentity:output:0*L
_input_shapes;
9:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_nameresult_grads_1:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
ù
e
I__inference_activation_3_layer_call_and_return_conditional_losses_4354218

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙(:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs
Ş
J
.__inference_activation_3_layer_call_fn_4354920

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_4354218`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙(:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs
Î	
p
$__inference_internal_grad_fn_4355169
result_grads_0
result_grads_1
sigmoid_inputs
identitye
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@U
mulMulsigmoid_inputssub:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?W
addAddV2add/x:output:0mul:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@T
mul_1MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Y
mul_2Mulresult_grads_0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Q
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
identityIdentity:output:0*L
_input_shapes;
9:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_nameresult_grads_1:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
÷9
ĵ
"__inference__wrapped_model_4354086
dense_inputB
/sequential_dense_matmul_readvariableop_resource:	(?
0sequential_dense_biasadd_readvariableop_resource:	E
1sequential_dense_1_matmul_readvariableop_resource:
A
2sequential_dense_1_biasadd_readvariableop_resource:	D
1sequential_dense_2_matmul_readvariableop_resource:	@@
2sequential_dense_2_biasadd_readvariableop_resource:@C
1sequential_dense_3_matmul_readvariableop_resource:@(@
2sequential_dense_3_biasadd_readvariableop_resource:(
identity˘'sequential/dense/BiasAdd/ReadVariableOp˘&sequential/dense/MatMul/ReadVariableOp˘)sequential/dense_1/BiasAdd/ReadVariableOp˘(sequential/dense_1/MatMul/ReadVariableOp˘)sequential/dense_2/BiasAdd/ReadVariableOp˘(sequential/dense_2/MatMul/ReadVariableOp˘)sequential/dense_3/BiasAdd/ReadVariableOp˘(sequential/dense_3/MatMul/ReadVariableOp
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype0
sequential/dense/MatMulMatMuldense_input.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ş
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
sequential/activation/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
sequential/activation/mulMul!sequential/dense/BiasAdd:output:0!sequential/activation/Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
sequential/activation/IdentityIdentitysequential/activation/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙è
sequential/activation/IdentityN	IdentityNsequential/activation/mul:z:0!sequential/dense/BiasAdd:output:0*
T
2*-
_gradient_op_typeCustomGradient-4354046*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
sequential/dropout/IdentityIdentity(sequential/activation/IdentityN:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0?
sequential/dense_1/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
sequential/activation_1/SigmoidSigmoid#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
sequential/activation_1/mulMul#sequential/dense_1/BiasAdd:output:0#sequential/activation_1/Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 sequential/activation_1/IdentityIdentitysequential/activation_1/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙î
!sequential/activation_1/IdentityN	IdentityNsequential/activation_1/mul:z:0#sequential/dense_1/BiasAdd:output:0*
T
2*-
_gradient_op_typeCustomGradient-4354059*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
sequential/dropout_1/IdentityIdentity*sequential/activation_1/IdentityN:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0Ż
sequential/dense_2/MatMulMatMul&sequential/dropout_1/Identity:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ż
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
sequential/activation_2/SigmoidSigmoid#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
sequential/activation_2/mulMul#sequential/dense_2/BiasAdd:output:0#sequential/activation_2/Sigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 sequential/activation_2/IdentityIdentitysequential/activation_2/mul:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@ì
!sequential/activation_2/IdentityN	IdentityNsequential/activation_2/mul:z:0#sequential/dense_2/BiasAdd:output:0*
T
2*-
_gradient_op_typeCustomGradient-4354072*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@
sequential/dropout_2/IdentityIdentity*sequential/activation_2/IdentityN:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(sequential/dense_3/MatMul/ReadVariableOpReadVariableOp1sequential_dense_3_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0Ż
sequential/dense_3/MatMulMatMul&sequential/dropout_2/Identity:output:00sequential/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0Ż
sequential/dense_3/BiasAddBiasAdd#sequential/dense_3/MatMul:product:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(r
IdentityIdentity#sequential/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp)^sequential/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙(: : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2T
(sequential/dense_3/MatMul/ReadVariableOp(sequential/dense_3/MatMul/ReadVariableOp:T P
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
%
_user_specified_namedense_input
Ħ
g
I__inference_activation_2_layer_call_and_return_conditional_losses_4354189

inputs

identity_1L
SigmoidSigmoidinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Q
mulMulinputsSigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@O
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
	IdentityN	IdentityNmul:z:0inputs*
T
2*-
_gradient_op_typeCustomGradient-4354182*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@\

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙@:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ö
b
)__inference_dropout_layer_call_fn_4354757

inputs
identity˘StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_4354354p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ö	
y
$__inference_internal_grad_fn_4355064
result_grads_0
result_grads_1
sigmoid_dense_1_biasadd
identityo
SigmoidSigmoidsigmoid_dense_1_biasadd^result_grads_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
mulMulsigmoid_dense_1_biasaddsub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
addAddV2add/x:output:0mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙U
mul_1MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_2Mulresult_grads_0	mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
IdentityIdentity	mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*O
_input_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:X T
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_1:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü	
Ĉ
,__inference_sequential_layer_call_fn_4354240
dense_input
unknown:	(
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	@
	unknown_4:@
	unknown_5:@(
	unknown_6:(
identity˘StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4354221o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙(: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
%
_user_specified_namedense_input
½.
ċ
G__inference_sequential_layer_call_and_return_conditional_losses_4354529
dense_input 
dense_4354501:	(
dense_4354503:	#
dense_1_4354508:

dense_1_4354510:	"
dense_2_4354515:	@
dense_2_4354517:@!
dense_3_4354522:@(
dense_3_4354524:(
identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall˘dense_3/StatefulPartitionedCall˘dropout/StatefulPartitionedCall˘!dropout_1/StatefulPartitionedCall˘!dropout_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_4354501dense_4354503*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_4354103á
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_4354119è
dropout/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_4354354
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_4354508dense_1_4354510*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_4354138ç
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_4354154
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_4354315
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_4354515dense_2_4354517*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_4354173ĉ
activation_2/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_4354189
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_4354276
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_3_4354522dense_3_4354524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_4354208ĉ
activation_3/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_4354218t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(ĥ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙(: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:T P
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
%
_user_specified_namedense_input
*
û
G__inference_sequential_layer_call_and_return_conditional_losses_4354498
dense_input 
dense_4354470:	(
dense_4354472:	#
dense_1_4354477:

dense_1_4354479:	"
dense_2_4354484:	@
dense_2_4354486:@!
dense_3_4354491:@(
dense_3_4354493:(
identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall˘dense_3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_4354470dense_4354472*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_4354103á
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_4354119Ĝ
dropout/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_4354126
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_4354477dense_1_4354479*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_4354138ç
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_4354154Ŝ
dropout_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_4354161
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_4354484dense_2_4354486*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_4354173ĉ
activation_2/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_4354189Ŭ
dropout_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_4354196
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_3_4354491dense_3_4354493*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_4354208ĉ
activation_3/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_4354218t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(Ì
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙(: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:T P
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
%
_user_specified_namedense_input
?.
à
G__inference_sequential_layer_call_and_return_conditional_losses_4354427

inputs 
dense_4354399:	(
dense_4354401:	#
dense_1_4354406:

dense_1_4354408:	"
dense_2_4354413:	@
dense_2_4354415:@!
dense_3_4354420:@(
dense_3_4354422:(
identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall˘dense_3/StatefulPartitionedCall˘dropout/StatefulPartitionedCall˘!dropout_1/StatefulPartitionedCall˘!dropout_2/StatefulPartitionedCallë
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4354399dense_4354401*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_4354103á
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_4354119è
dropout/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_4354354
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_4354406dense_1_4354408*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_4354138ç
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_4354154
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_4354315
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_4354413dense_2_4354415*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_4354173ĉ
activation_2/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_4354189
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_4354276
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_3_4354420dense_3_4354422*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_4354208ĉ
activation_3/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_4354218t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(ĥ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙(: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs
Ù
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_4354884

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙@:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ŭ
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_4354823

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ù
e
I__inference_activation_3_layer_call_and_return_conditional_losses_4354924

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙(:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs
Ş
J
.__inference_activation_2_layer_call_fn_4354859

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_4354189`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙@:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
0
Ĵ
G__inference_sequential_layer_call_and_return_conditional_losses_4354643

inputs7
$dense_matmul_readvariableop_resource:	(4
%dense_biasadd_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	9
&dense_2_matmul_readvariableop_resource:	@5
'dense_2_biasadd_readvariableop_resource:@8
&dense_3_matmul_readvariableop_resource:@(5
'dense_3_biasadd_readvariableop_resource:(
identity˘dense/BiasAdd/ReadVariableOp˘dense/MatMul/ReadVariableOp˘dense_1/BiasAdd/ReadVariableOp˘dense_1/MatMul/ReadVariableOp˘dense_2/BiasAdd/ReadVariableOp˘dense_2/MatMul/ReadVariableOp˘dense_3/BiasAdd/ReadVariableOp˘dense_3/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
activation/SigmoidSigmoiddense/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
activation/mulMuldense/BiasAdd:output:0activation/Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
activation/IdentityIdentityactivation/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ç
activation/IdentityN	IdentityNactivation/mul:z:0dense/BiasAdd:output:0*
T
2*-
_gradient_op_typeCustomGradient-4354603*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙n
dropout/IdentityIdentityactivation/IdentityN:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
activation_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
activation_1/mulMuldense_1/BiasAdd:output:0activation_1/Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
activation_1/IdentityIdentityactivation_1/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Í
activation_1/IdentityN	IdentityNactivation_1/mul:z:0dense_1/BiasAdd:output:0*
T
2*-
_gradient_op_typeCustomGradient-4354616*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙r
dropout_1/IdentityIdentityactivation_1/IdentityN:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_2/MatMulMatMuldropout_1/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@k
activation_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@}
activation_2/mulMuldense_2/BiasAdd:output:0activation_2/Sigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@i
activation_2/IdentityIdentityactivation_2/mul:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Ë
activation_2/IdentityN	IdentityNactivation_2/mul:z:0dense_2/BiasAdd:output:0*
T
2*-
_gradient_op_typeCustomGradient-4354629*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@q
dropout_2/IdentityIdentityactivation_2/IdentityN:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0
dense_3/MatMulMatMuldropout_2/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(Ĉ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙(: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs

Ù
 __inference__traced_save_4355196
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Â
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*ë
valueáBŜ	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*]
_input_shapesL
J: :	(::
::	@:@:@(:(: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	(:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@(: 

_output_shapes
:(:	

_output_shapes
: 
Ë	
ö
D__inference_dense_2_layer_call_and_return_conditional_losses_4354173

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ħ
g
I__inference_activation_2_layer_call_and_return_conditional_losses_4354869

inputs

identity_1L
SigmoidSigmoidinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Q
mulMulinputsSigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@O
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
	IdentityN	IdentityNmul:z:0inputs*
T
2*-
_gradient_op_typeCustomGradient-4354862*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@\

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙@:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
§
e
G__inference_activation_layer_call_and_return_conditional_losses_4354747

inputs

identity_1M
SigmoidSigmoidinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
mulMulinputsSigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙P
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ħ
	IdentityN	IdentityNmul:z:0inputs*
T
2*-
_gradient_op_typeCustomGradient-4354740*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙]

Identity_1IdentityIdentityN:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
?
J
.__inference_activation_1_layer_call_fn_4354798

inputs
identity¸
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_4354154a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ô	
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_4354276

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ĥ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙@:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
?	
w
$__inference_internal_grad_fn_4355094
result_grads_0
result_grads_1
sigmoid_dense_biasadd
identitym
SigmoidSigmoidsigmoid_dense_biasadd^result_grads_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
mulMulsigmoid_dense_biasaddsub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
addAddV2add/x:output:0mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙U
mul_1MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_2Mulresult_grads_0	mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
IdentityIdentity	mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*O
_input_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:X T
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_1:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Û	
p
$__inference_internal_grad_fn_4355154
result_grads_0
result_grads_1
sigmoid_inputs
identityf
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mulMulsigmoid_inputssub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
addAddV2add/x:output:0mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙U
mul_1MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_2Mulresult_grads_0	mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
IdentityIdentity	mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*O
_input_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:X T
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_1:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ċ

)__inference_dense_3_layer_call_fn_4354905

inputs
unknown:@(
	unknown_0:(
identity˘StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_4354208o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
§
e
G__inference_activation_layer_call_and_return_conditional_losses_4354119

inputs

identity_1M
SigmoidSigmoidinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
mulMulinputsSigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙P
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ħ
	IdentityN	IdentityNmul:z:0inputs*
T
2*-
_gradient_op_typeCustomGradient-4354112*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙]

Identity_1IdentityIdentityN:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ĝ)
ö
G__inference_sequential_layer_call_and_return_conditional_losses_4354221

inputs 
dense_4354104:	(
dense_4354106:	#
dense_1_4354139:

dense_1_4354141:	"
dense_2_4354174:	@
dense_2_4354176:@!
dense_3_4354209:@(
dense_3_4354211:(
identity˘dense/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall˘dense_3/StatefulPartitionedCallë
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4354104dense_4354106*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_4354103á
activation/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_4354119Ĝ
dropout/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_4354126
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_4354139dense_1_4354141*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_4354138ç
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_4354154Ŝ
dropout_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_4354161
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_4354174dense_2_4354176*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_4354173ĉ
activation_2/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_4354189Ŭ
dropout_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_4354196
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_3_4354209dense_3_4354211*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_4354208ĉ
activation_3/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_4354218t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(Ì
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙(: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs
ú	
c
D__inference_dropout_layer_call_and_return_conditional_losses_4354774

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ç	
ġ
D__inference_dense_3_layer_call_and_return_conditional_losses_4354208

inputs0
matmul_readvariableop_resource:@(-
biasadd_readvariableop_resource:(
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Í	
Á
,__inference_sequential_layer_call_fn_4354594

inputs
unknown:	(
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	@
	unknown_4:@
	unknown_5:@(
	unknown_6:(
identity˘StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4354427o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙(: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs
İ
g
I__inference_activation_1_layer_call_and_return_conditional_losses_4354154

inputs

identity_1M
SigmoidSigmoidinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
mulMulinputsSigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙P
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ħ
	IdentityN	IdentityNmul:z:0inputs*
T
2*-
_gradient_op_typeCustomGradient-4354147*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙]

Identity_1IdentityIdentityN:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ü	
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_4354315

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ċ

'__inference_dense_layer_call_fn_4354722

inputs
unknown:	(
	unknown_0:	
identity˘StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_4354103p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs
Í	
Á
,__inference_sequential_layer_call_fn_4354573

inputs
unknown:	(
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	@
	unknown_4:@
	unknown_5:@(
	unknown_6:(
identity˘StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4354221o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙(: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs
Ü	
Ĉ
,__inference_sequential_layer_call_fn_4354467
dense_input
unknown:	(
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	@
	unknown_4:@
	unknown_5:@(
	unknown_6:(
identity˘StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4354427o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙(: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
%
_user_specified_namedense_input
Ş
H
,__inference_activation_layer_call_fn_4354737

inputs
identityĥ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_4354119a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ú
d
+__inference_dropout_1_layer_call_fn_4354818

inputs
identity˘StatefulPartitionedCallĊ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_4354315p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Û	
p
$__inference_internal_grad_fn_4355019
result_grads_0
result_grads_1
sigmoid_inputs
identityf
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mulMulsigmoid_inputssub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
addAddV2add/x:output:0mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙U
mul_1MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_2Mulresult_grads_0	mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
IdentityIdentity	mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*O
_input_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:X T
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_1:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ö	
y
$__inference_internal_grad_fn_4355109
result_grads_0
result_grads_1
sigmoid_dense_1_biasadd
identityo
SigmoidSigmoidsigmoid_dense_1_biasadd^result_grads_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙_
mulMulsigmoid_dense_1_biasaddsub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
addAddV2add/x:output:0mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙U
mul_1MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_2Mulresult_grads_0	mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
IdentityIdentity	mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*O
_input_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:X T
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_1:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ÈF
Ĵ
G__inference_sequential_layer_call_and_return_conditional_losses_4354713

inputs7
$dense_matmul_readvariableop_resource:	(4
%dense_biasadd_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	9
&dense_2_matmul_readvariableop_resource:	@5
'dense_2_biasadd_readvariableop_resource:@8
&dense_3_matmul_readvariableop_resource:@(5
'dense_3_biasadd_readvariableop_resource:(
identity˘dense/BiasAdd/ReadVariableOp˘dense/MatMul/ReadVariableOp˘dense_1/BiasAdd/ReadVariableOp˘dense_1/MatMul/ReadVariableOp˘dense_2/BiasAdd/ReadVariableOp˘dense_2/MatMul/ReadVariableOp˘dense_3/BiasAdd/ReadVariableOp˘dense_3/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
activation/SigmoidSigmoiddense/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
activation/mulMuldense/BiasAdd:output:0activation/Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
activation/IdentityIdentityactivation/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ç
activation/IdentityN	IdentityNactivation/mul:z:0dense/BiasAdd:output:0*
T
2*-
_gradient_op_typeCustomGradient-4354652*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout/dropout/MulMulactivation/IdentityN:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
dropout/dropout/ShapeShapeactivation/IdentityN:output:0*
T0*
_output_shapes
:
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ż
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
activation_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
activation_1/mulMuldense_1/BiasAdd:output:0activation_1/Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
activation_1/IdentityIdentityactivation_1/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Í
activation_1/IdentityN	IdentityNactivation_1/mul:z:0dense_1/BiasAdd:output:0*
T
2*-
_gradient_op_typeCustomGradient-4354672*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_1/dropout/MulMulactivation_1/IdentityN:output:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
dropout_1/dropout/ShapeShapeactivation_1/IdentityN:output:0*
T0*
_output_shapes
:Ħ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ċ
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@k
activation_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@}
activation_2/mulMuldense_2/BiasAdd:output:0activation_2/Sigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@i
activation_2/IdentityIdentityactivation_2/mul:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Ë
activation_2/IdentityN	IdentityNactivation_2/mul:z:0dense_2/BiasAdd:output:0*
T
2*-
_gradient_op_typeCustomGradient-4354692*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_2/dropout/MulMulactivation_2/IdentityN:output:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@f
dropout_2/dropout/ShapeShapeactivation_2/IdentityN:output:0*
T0*
_output_shapes
: 
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ä
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@(*
dtype0
dense_3/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(Ĉ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙(: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs
ü	
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_4354835

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ò	
ĝ
D__inference_dense_1_layer_call_and_return_conditional_losses_4354793

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs



$__inference_internal_grad_fn_4354959
result_grads_0
result_grads_1$
 sigmoid_sequential_dense_biasadd
identityx
SigmoidSigmoid sigmoid_sequential_dense_biasadd^result_grads_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
mulMul sigmoid_sequential_dense_biasaddsub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
addAddV2add/x:output:0mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙U
mul_1MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_2Mulresult_grads_0	mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
IdentityIdentity	mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*O
_input_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:X T
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_1:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Û	
p
$__inference_internal_grad_fn_4355004
result_grads_0
result_grads_1
sigmoid_inputs
identityf
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mulMulsigmoid_inputssub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
addAddV2add/x:output:0mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙U
mul_1MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_2Mulresult_grads_0	mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
IdentityIdentity	mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*O
_input_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:X T
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_1:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
°	
ż
%__inference_signature_wrapper_4354552
dense_input
unknown:	(
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	@
	unknown_4:@
	unknown_5:@(
	unknown_6:(
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙(**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_4354086o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙(: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
%
_user_specified_namedense_input
İ
g
I__inference_activation_1_layer_call_and_return_conditional_losses_4354808

inputs

identity_1M
SigmoidSigmoidinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
mulMulinputsSigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙P
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ħ
	IdentityN	IdentityNmul:z:0inputs*
T
2*-
_gradient_op_typeCustomGradient-4354801*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙]

Identity_1IdentityIdentityN:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ì	
ġ
B__inference_dense_layer_call_and_return_conditional_losses_4354732

inputs1
matmul_readvariableop_resource:	(.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs
ô	
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_4354896

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ĥ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙@:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs



$__inference_internal_grad_fn_4354989
result_grads_0
result_grads_1&
"sigmoid_sequential_dense_2_biasadd
identityy
SigmoidSigmoid"sigmoid_sequential_dense_2_biasadd^result_grads_0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@i
mulMul"sigmoid_sequential_dense_2_biasaddsub:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?W
addAddV2add/x:output:0mul:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@T
mul_1MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Y
mul_2Mulresult_grads_0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Q
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
identityIdentity:output:0*L
_input_shapes;
9:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_nameresult_grads_1:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Ŭ
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_4354161

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
?	
w
$__inference_internal_grad_fn_4355049
result_grads_0
result_grads_1
sigmoid_dense_biasadd
identitym
SigmoidSigmoidsigmoid_dense_biasadd^result_grads_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
mulMulsigmoid_dense_biasaddsub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
addAddV2add/x:output:0mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙U
mul_1MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_2Mulresult_grads_0	mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
IdentityIdentity	mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*O
_input_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:X T
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_1:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ö
d
+__inference_dropout_2_layer_call_fn_4354879

inputs
identity˘StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_4354276o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
¤
E
)__inference_dropout_layer_call_fn_4354752

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_4354126a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ì

)__inference_dense_1_layer_call_fn_4354783

inputs
unknown:

	unknown_0:	
identity˘StatefulPartitionedCallŬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_4354138p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
È

)__inference_dense_2_layer_call_fn_4354844

inputs
unknown:	@
	unknown_0:@
identity˘StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_4354173o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ò#
˙
#__inference__traced_restore_4355230
file_prefix0
assignvariableop_dense_kernel:	(,
assignvariableop_1_dense_bias:	5
!assignvariableop_2_dense_1_kernel:
.
assignvariableop_3_dense_1_bias:	4
!assignvariableop_4_dense_2_kernel:	@-
assignvariableop_5_dense_2_bias:@3
!assignvariableop_6_dense_3_kernel:@(-
assignvariableop_7_dense_3_bias:(

identity_9˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7Ċ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*ë
valueáBŜ	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B Ë
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_9IdentityIdentity_8:output:0^NoOp_1*
T0*
_output_shapes
: î
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*"
_acd_function_control_output(*
_output_shapes
 "!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ù
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_4354196

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:˙˙˙˙˙˙˙˙˙@:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
é	
y
$__inference_internal_grad_fn_4355079
result_grads_0
result_grads_1
sigmoid_dense_2_biasadd
identityn
SigmoidSigmoidsigmoid_dense_2_biasadd^result_grads_0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@^
mulMulsigmoid_dense_2_biasaddsub:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?W
addAddV2add/x:output:0mul:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@T
mul_1MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Y
mul_2Mulresult_grads_0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Q
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
identityIdentity:output:0*L
_input_shapes;
9:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_nameresult_grads_1:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Û	
p
$__inference_internal_grad_fn_4355139
result_grads_0
result_grads_1
sigmoid_inputs
identityf
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mulMulsigmoid_inputssub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
addAddV2add/x:output:0mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙U
mul_1MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_2Mulresult_grads_0	mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
IdentityIdentity	mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*O
_input_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:X T
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_nameresult_grads_1:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
é	
y
$__inference_internal_grad_fn_4355124
result_grads_0
result_grads_1
sigmoid_dense_2_biasadd
identityn
SigmoidSigmoidsigmoid_dense_2_biasadd^result_grads_0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@^
mulMulsigmoid_dense_2_biasaddsub:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?W
addAddV2add/x:output:0mul:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@T
mul_1MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Y
mul_2Mulresult_grads_0	mul_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@Q
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
identityIdentity:output:0*L
_input_shapes;
9:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@:˙˙˙˙˙˙˙˙˙@:W S
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_nameresult_grads_1:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Ì	
ġ
B__inference_dense_layer_call_and_return_conditional_losses_4354103

inputs1
matmul_readvariableop_resource:	(.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs
ú	
c
D__inference_dropout_layer_call_and_return_conditional_losses_4354354

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¨
G
+__inference_dropout_1_layer_call_fn_4354813

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_4354161a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs>
$__inference_internal_grad_fn_4354959CustomGradient-4354046>
$__inference_internal_grad_fn_4354974CustomGradient-4354059>
$__inference_internal_grad_fn_4354989CustomGradient-4354072>
$__inference_internal_grad_fn_4355004CustomGradient-4354112>
$__inference_internal_grad_fn_4355019CustomGradient-4354147>
$__inference_internal_grad_fn_4355034CustomGradient-4354182>
$__inference_internal_grad_fn_4355049CustomGradient-4354603>
$__inference_internal_grad_fn_4355064CustomGradient-4354616>
$__inference_internal_grad_fn_4355079CustomGradient-4354629>
$__inference_internal_grad_fn_4355094CustomGradient-4354652>
$__inference_internal_grad_fn_4355109CustomGradient-4354672>
$__inference_internal_grad_fn_4355124CustomGradient-4354692>
$__inference_internal_grad_fn_4355139CustomGradient-4354740>
$__inference_internal_grad_fn_4355154CustomGradient-4354801>
$__inference_internal_grad_fn_4355169CustomGradient-4354862"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*·
serving_default£
C
dense_input4
serving_default_dense_input:0˙˙˙˙˙˙˙˙˙(@
activation_30
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙(tensorflow/serving/predict:ÀÑ
Ò
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
	variables
trainable_variables
regularization_losses
	keras_api

signatures
__call__
+&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_sequential
½

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
§
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
§
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
½

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
§
%	variables
&trainable_variables
'regularization_losses
(	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
§
)	variables
*trainable_variables
+regularization_losses
,	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
½

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
§
3	variables
4trainable_variables
5regularization_losses
6	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
§
7	variables
8trainable_variables
9regularization_losses
:	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
½

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
§
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
X
0
1
2
 3
-4
.5
;6
<7"
trackable_list_wrapper
X
0
1
2
 3
-4
.5
;6
<7"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
:	(2dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_1/kernel
:2dense_1/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
!	variables
"trainable_variables
#regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
%	variables
&trainable_variables
'regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
)	variables
*trainable_variables
+regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:	@2dense_2/kernel
:@2dense_2/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
/	variables
0trainable_variables
1regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
3	variables
4trainable_variables
5regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
7	variables
8trainable_variables
9regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 :@(2dense_3/kernel
:(2dense_3/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
=	variables
>trainable_variables
?regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ħ
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŝ2û
,__inference_sequential_layer_call_fn_4354240
,__inference_sequential_layer_call_fn_4354573
,__inference_sequential_layer_call_fn_4354594
,__inference_sequential_layer_call_fn_4354467À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ê2ç
G__inference_sequential_layer_call_and_return_conditional_losses_4354643
G__inference_sequential_layer_call_and_return_conditional_losses_4354713
G__inference_sequential_layer_call_and_return_conditional_losses_4354498
G__inference_sequential_layer_call_and_return_conditional_losses_4354529À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ÑBÎ
"__inference__wrapped_model_4354086dense_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ñ2Î
'__inference_dense_layer_call_fn_4354722˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ì2é
B__inference_dense_layer_call_and_return_conditional_losses_4354732˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ö2Ó
,__inference_activation_layer_call_fn_4354737˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ñ2î
G__inference_activation_layer_call_and_return_conditional_losses_4354747˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
)__inference_dropout_layer_call_fn_4354752
)__inference_dropout_layer_call_fn_4354757´
Ğ²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ĉ2?
D__inference_dropout_layer_call_and_return_conditional_losses_4354762
D__inference_dropout_layer_call_and_return_conditional_losses_4354774´
Ğ²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ó2?
)__inference_dense_1_layer_call_fn_4354783˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
î2ë
D__inference_dense_1_layer_call_and_return_conditional_losses_4354793˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ĝ2Ġ
.__inference_activation_1_layer_call_fn_4354798˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ó2?
I__inference_activation_1_layer_call_and_return_conditional_losses_4354808˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
+__inference_dropout_1_layer_call_fn_4354813
+__inference_dropout_1_layer_call_fn_4354818´
Ğ²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ê2Ç
F__inference_dropout_1_layer_call_and_return_conditional_losses_4354823
F__inference_dropout_1_layer_call_and_return_conditional_losses_4354835´
Ğ²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ó2?
)__inference_dense_2_layer_call_fn_4354844˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
î2ë
D__inference_dense_2_layer_call_and_return_conditional_losses_4354854˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ĝ2Ġ
.__inference_activation_2_layer_call_fn_4354859˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ó2?
I__inference_activation_2_layer_call_and_return_conditional_losses_4354869˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
+__inference_dropout_2_layer_call_fn_4354874
+__inference_dropout_2_layer_call_fn_4354879´
Ğ²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ê2Ç
F__inference_dropout_2_layer_call_and_return_conditional_losses_4354884
F__inference_dropout_2_layer_call_and_return_conditional_losses_4354896´
Ğ²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ó2?
)__inference_dense_3_layer_call_fn_4354905˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
î2ë
D__inference_dense_3_layer_call_and_return_conditional_losses_4354915˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ĝ2Ġ
.__inference_activation_3_layer_call_fn_4354920˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ó2?
I__inference_activation_3_layer_call_and_return_conditional_losses_4354924˘
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
?BÍ
%__inference_signature_wrapper_4354552dense_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Bb@
sequential/dense/BiasAdd:0"__inference__wrapped_model_4354086
DbB
sequential/dense_1/BiasAdd:0"__inference__wrapped_model_4354086
DbB
sequential/dense_2/BiasAdd:0"__inference__wrapped_model_4354086
UbS
inputs:0G__inference_activation_layer_call_and_return_conditional_losses_4354119
WbU
inputs:0I__inference_activation_1_layer_call_and_return_conditional_losses_4354154
WbU
inputs:0I__inference_activation_2_layer_call_and_return_conditional_losses_4354189
\bZ
dense/BiasAdd:0G__inference_sequential_layer_call_and_return_conditional_losses_4354643
^b\
dense_1/BiasAdd:0G__inference_sequential_layer_call_and_return_conditional_losses_4354643
^b\
dense_2/BiasAdd:0G__inference_sequential_layer_call_and_return_conditional_losses_4354643
\bZ
dense/BiasAdd:0G__inference_sequential_layer_call_and_return_conditional_losses_4354713
^b\
dense_1/BiasAdd:0G__inference_sequential_layer_call_and_return_conditional_losses_4354713
^b\
dense_2/BiasAdd:0G__inference_sequential_layer_call_and_return_conditional_losses_4354713
UbS
inputs:0G__inference_activation_layer_call_and_return_conditional_losses_4354747
WbU
inputs:0I__inference_activation_1_layer_call_and_return_conditional_losses_4354808
WbU
inputs:0I__inference_activation_2_layer_call_and_return_conditional_losses_4354869£
"__inference__wrapped_model_4354086} -.;<4˘1
*˘'
%"
dense_input˙˙˙˙˙˙˙˙˙(
Ş ";Ş8
6
activation_3&#
activation_3˙˙˙˙˙˙˙˙˙(§
I__inference_activation_1_layer_call_and_return_conditional_losses_4354808Z0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
.__inference_activation_1_layer_call_fn_4354798M0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙?
I__inference_activation_2_layer_call_and_return_conditional_losses_4354869X/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙@
Ş "%˘"

0˙˙˙˙˙˙˙˙˙@
 }
.__inference_activation_2_layer_call_fn_4354859K/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙@
Ş "˙˙˙˙˙˙˙˙˙@?
I__inference_activation_3_layer_call_and_return_conditional_losses_4354924X/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙(
Ş "%˘"

0˙˙˙˙˙˙˙˙˙(
 }
.__inference_activation_3_layer_call_fn_4354920K/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙(
Ş "˙˙˙˙˙˙˙˙˙(?
G__inference_activation_layer_call_and_return_conditional_losses_4354747Z0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 }
,__inference_activation_layer_call_fn_4354737M0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ĥ
D__inference_dense_1_layer_call_and_return_conditional_losses_4354793^ 0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ~
)__inference_dense_1_layer_call_fn_4354783Q 0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙?
D__inference_dense_2_layer_call_and_return_conditional_losses_4354854]-.0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙@
 }
)__inference_dense_2_layer_call_fn_4354844P-.0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙@¤
D__inference_dense_3_layer_call_and_return_conditional_losses_4354915\;</˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙@
Ş "%˘"

0˙˙˙˙˙˙˙˙˙(
 |
)__inference_dense_3_layer_call_fn_4354905O;</˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙@
Ş "˙˙˙˙˙˙˙˙˙(£
B__inference_dense_layer_call_and_return_conditional_losses_4354732]/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙(
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 {
'__inference_dense_layer_call_fn_4354722P/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙(
Ş "˙˙˙˙˙˙˙˙˙¨
F__inference_dropout_1_layer_call_and_return_conditional_losses_4354823^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ¨
F__inference_dropout_1_layer_call_and_return_conditional_losses_4354835^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
+__inference_dropout_1_layer_call_fn_4354813Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙
+__inference_dropout_1_layer_call_fn_4354818Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙Ĥ
F__inference_dropout_2_layer_call_and_return_conditional_losses_4354884\3˘0
)˘&
 
inputs˙˙˙˙˙˙˙˙˙@
p 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙@
 Ĥ
F__inference_dropout_2_layer_call_and_return_conditional_losses_4354896\3˘0
)˘&
 
inputs˙˙˙˙˙˙˙˙˙@
p
Ş "%˘"

0˙˙˙˙˙˙˙˙˙@
 ~
+__inference_dropout_2_layer_call_fn_4354874O3˘0
)˘&
 
inputs˙˙˙˙˙˙˙˙˙@
p 
Ş "˙˙˙˙˙˙˙˙˙@~
+__inference_dropout_2_layer_call_fn_4354879O3˘0
)˘&
 
inputs˙˙˙˙˙˙˙˙˙@
p
Ş "˙˙˙˙˙˙˙˙˙@Ĥ
D__inference_dropout_layer_call_and_return_conditional_losses_4354762^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 Ĥ
D__inference_dropout_layer_call_and_return_conditional_losses_4354774^4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ~
)__inference_dropout_layer_call_fn_4354752Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "˙˙˙˙˙˙˙˙˙~
)__inference_dropout_layer_call_fn_4354757Q4˘1
*˘'
!
inputs˙˙˙˙˙˙˙˙˙
p
Ş "˙˙˙˙˙˙˙˙˙½
$__inference_internal_grad_fn_4354959g˘d
]˘Z

 
)&
result_grads_0˙˙˙˙˙˙˙˙˙
)&
result_grads_1˙˙˙˙˙˙˙˙˙
Ş "%"

 

1˙˙˙˙˙˙˙˙˙½
$__inference_internal_grad_fn_4354974g˘d
]˘Z

 
)&
result_grads_0˙˙˙˙˙˙˙˙˙
)&
result_grads_1˙˙˙˙˙˙˙˙˙
Ş "%"

 

1˙˙˙˙˙˙˙˙˙ş
$__inference_internal_grad_fn_4354989e˘b
[˘X

 
(%
result_grads_0˙˙˙˙˙˙˙˙˙@
(%
result_grads_1˙˙˙˙˙˙˙˙˙@
Ş "$!

 

1˙˙˙˙˙˙˙˙˙@½
$__inference_internal_grad_fn_4355004g˘d
]˘Z

 
)&
result_grads_0˙˙˙˙˙˙˙˙˙
)&
result_grads_1˙˙˙˙˙˙˙˙˙
Ş "%"

 

1˙˙˙˙˙˙˙˙˙½
$__inference_internal_grad_fn_4355019g˘d
]˘Z

 
)&
result_grads_0˙˙˙˙˙˙˙˙˙
)&
result_grads_1˙˙˙˙˙˙˙˙˙
Ş "%"

 

1˙˙˙˙˙˙˙˙˙ş
$__inference_internal_grad_fn_4355034 e˘b
[˘X

 
(%
result_grads_0˙˙˙˙˙˙˙˙˙@
(%
result_grads_1˙˙˙˙˙˙˙˙˙@
Ş "$!

 

1˙˙˙˙˙˙˙˙˙@½
$__inference_internal_grad_fn_4355049Ħg˘d
]˘Z

 
)&
result_grads_0˙˙˙˙˙˙˙˙˙
)&
result_grads_1˙˙˙˙˙˙˙˙˙
Ş "%"

 

1˙˙˙˙˙˙˙˙˙½
$__inference_internal_grad_fn_4355064˘g˘d
]˘Z

 
)&
result_grads_0˙˙˙˙˙˙˙˙˙
)&
result_grads_1˙˙˙˙˙˙˙˙˙
Ş "%"

 

1˙˙˙˙˙˙˙˙˙ş
$__inference_internal_grad_fn_4355079£e˘b
[˘X

 
(%
result_grads_0˙˙˙˙˙˙˙˙˙@
(%
result_grads_1˙˙˙˙˙˙˙˙˙@
Ş "$!

 

1˙˙˙˙˙˙˙˙˙@½
$__inference_internal_grad_fn_4355094¤g˘d
]˘Z

 
)&
result_grads_0˙˙˙˙˙˙˙˙˙
)&
result_grads_1˙˙˙˙˙˙˙˙˙
Ş "%"

 

1˙˙˙˙˙˙˙˙˙½
$__inference_internal_grad_fn_4355109?g˘d
]˘Z

 
)&
result_grads_0˙˙˙˙˙˙˙˙˙
)&
result_grads_1˙˙˙˙˙˙˙˙˙
Ş "%"

 

1˙˙˙˙˙˙˙˙˙ş
$__inference_internal_grad_fn_4355124Ĥe˘b
[˘X

 
(%
result_grads_0˙˙˙˙˙˙˙˙˙@
(%
result_grads_1˙˙˙˙˙˙˙˙˙@
Ş "$!

 

1˙˙˙˙˙˙˙˙˙@½
$__inference_internal_grad_fn_4355139§g˘d
]˘Z

 
)&
result_grads_0˙˙˙˙˙˙˙˙˙
)&
result_grads_1˙˙˙˙˙˙˙˙˙
Ş "%"

 

1˙˙˙˙˙˙˙˙˙½
$__inference_internal_grad_fn_4355154¨g˘d
]˘Z

 
)&
result_grads_0˙˙˙˙˙˙˙˙˙
)&
result_grads_1˙˙˙˙˙˙˙˙˙
Ş "%"

 

1˙˙˙˙˙˙˙˙˙ş
$__inference_internal_grad_fn_4355169İe˘b
[˘X

 
(%
result_grads_0˙˙˙˙˙˙˙˙˙@
(%
result_grads_1˙˙˙˙˙˙˙˙˙@
Ş "$!

 

1˙˙˙˙˙˙˙˙˙@ş
G__inference_sequential_layer_call_and_return_conditional_losses_4354498o -.;<<˘9
2˘/
%"
dense_input˙˙˙˙˙˙˙˙˙(
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙(
 ş
G__inference_sequential_layer_call_and_return_conditional_losses_4354529o -.;<<˘9
2˘/
%"
dense_input˙˙˙˙˙˙˙˙˙(
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙(
 µ
G__inference_sequential_layer_call_and_return_conditional_losses_4354643j -.;<7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙(
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙(
 µ
G__inference_sequential_layer_call_and_return_conditional_losses_4354713j -.;<7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙(
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙(
 
,__inference_sequential_layer_call_fn_4354240b -.;<<˘9
2˘/
%"
dense_input˙˙˙˙˙˙˙˙˙(
p 

 
Ş "˙˙˙˙˙˙˙˙˙(
,__inference_sequential_layer_call_fn_4354467b -.;<<˘9
2˘/
%"
dense_input˙˙˙˙˙˙˙˙˙(
p

 
Ş "˙˙˙˙˙˙˙˙˙(
,__inference_sequential_layer_call_fn_4354573] -.;<7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙(
p 

 
Ş "˙˙˙˙˙˙˙˙˙(
,__inference_sequential_layer_call_fn_4354594] -.;<7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙(
p

 
Ş "˙˙˙˙˙˙˙˙˙(ĥ
%__inference_signature_wrapper_4354552 -.;<C˘@
˘ 
9Ş6
4
dense_input%"
dense_input˙˙˙˙˙˙˙˙˙(";Ş8
6
activation_3&#
activation_3˙˙˙˙˙˙˙˙˙(