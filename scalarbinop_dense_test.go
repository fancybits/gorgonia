package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

// These tests cover scalarBinOp.Do() being handed 0-d (scalar-shaped)
// *tensor.Dense values instead of boxed Scalar values (*F64, *F32, ...).
//
// NodeFromAny on a 0-d tensor.Tensor produces a *Node whose static type is
// a scalar tensor.Dtype (see NodeFromAny/NewTensor with dims==0), which
// causes newEBOByType to select scalarBinOp - but the runtime Value
// attached to the node remains the original *tensor.Dense, unboxed. This
// mirrors how onnx-go's gorgonnx backend feeds 0-d tensors into Lt/Gt.
func denseScalar(t *testing.T, dt tensor.Dtype, backing interface{}) tensor.Tensor {
	t.Helper()
	return tensor.New(tensor.Of(dt), tensor.WithShape(), tensor.WithBacking(backing))
}

func runScalarBinOp(t *testing.T, op ʘBinaryOperatorType, retSame bool, a, b tensor.Tensor) Value {
	t.Helper()
	g := NewGraph()
	na := NodeFromAny(g, a, WithName("a"))
	nb := NodeFromAny(g, b, WithName("b"))

	var ret *Node
	var err error
	switch op {
	case ltOpType:
		ret, err = Lt(na, nb, retSame)
	case gtOpType:
		ret, err = Gt(na, nb, retSame)
	case addOpType:
		ret, err = Add(na, nb)
	default:
		t.Fatalf("unsupported op %v in test helper", op)
	}
	require.NoError(t, err)

	m := NewTapeMachine(g)
	defer m.Close()
	require.NoError(t, m.RunAll())

	return ret.Value()
}

func TestScalarBinOp_DenseLt_Float32(t *testing.T) {
	tests := []struct {
		name     string
		a, b     float32
		expected bool
	}{
		{"true", 1, 2, true},
		{"false", 2, 1, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := denseScalar(t, tensor.Float32, []float32{tc.a})
			b := denseScalar(t, tensor.Float32, []float32{tc.b})

			got := runScalarBinOp(t, ltOpType, false, a, b)
			bv, ok := got.(*B)
			require.True(t, ok, "expected *B, got %T", got)
			assert.Equal(t, tc.expected, bool(*bv))
		})
	}
}

func TestScalarBinOp_DenseGt_Float32(t *testing.T) {
	tests := []struct {
		name     string
		a, b     float32
		expected bool
	}{
		{"true", 2, 1, true},
		{"false", 1, 2, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := denseScalar(t, tensor.Float32, []float32{tc.a})
			b := denseScalar(t, tensor.Float32, []float32{tc.b})

			got := runScalarBinOp(t, gtOpType, false, a, b)
			bv, ok := got.(*B)
			require.True(t, ok, "expected *B, got %T", got)
			assert.Equal(t, tc.expected, bool(*bv))
		})
	}
}

func TestScalarBinOp_DenseAdd_Float32(t *testing.T) {
	a := denseScalar(t, tensor.Float32, []float32{1.5})
	b := denseScalar(t, tensor.Float32, []float32{2.5})

	got := runScalarBinOp(t, addOpType, false, a, b)
	fv, ok := got.(*F32)
	require.True(t, ok, "expected *F32, got %T", got)
	assert.Equal(t, float32(4.0), float32(*fv))
}

func TestScalarBinOp_DenseAdd_Int32(t *testing.T) {
	a := denseScalar(t, tensor.Int32, []int32{3})
	b := denseScalar(t, tensor.Int32, []int32{4})

	got := runScalarBinOp(t, addOpType, false, a, b)
	iv, ok := got.(*I32)
	require.True(t, ok, "expected *I32, got %T", got)
	assert.Equal(t, int32(7), int32(*iv))
}

// TestScalarBinOp_DenseLt_RetSame covers the retSame=true path for a
// comparison op fed 0-d *tensor.Dense operands: the boolean result must be
// converted back to the operand dtype (1.0/0.0), not left as *B.
func TestScalarBinOp_DenseLt_RetSame(t *testing.T) {
	a := denseScalar(t, tensor.Float32, []float32{1})
	b := denseScalar(t, tensor.Float32, []float32{2})

	got := runScalarBinOp(t, ltOpType, true, a, b)
	fv, ok := got.(*F32)
	require.True(t, ok, "expected *F32, got %T", got)
	assert.Equal(t, float32(1.0), float32(*fv))

	a = denseScalar(t, tensor.Float32, []float32{2})
	b = denseScalar(t, tensor.Float32, []float32{1})

	got = runScalarBinOp(t, ltOpType, true, a, b)
	fv, ok = got.(*F32)
	require.True(t, ok, "expected *F32, got %T", got)
	assert.Equal(t, float32(0.0), float32(*fv))
}
