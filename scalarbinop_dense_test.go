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

// asDenseScalar asserts got is a 0-d *tensor.Dense and returns its
// ScalarValue(). scalarBinOp.Do keeps Dense values Dense: when either
// input arrived as a 0-d *tensor.Dense, the result is re-boxed as a 0-d
// *tensor.Dense too (rather than a boxed Scalar), so downstream consumers
// that only handle *tensor.Dense (like onnx-go's gorgonnx ops) keep working
// on comparison/arithmetic results the same way they do on their inputs.
func asDenseScalar(t *testing.T, got Value) interface{} {
	t.Helper()
	d, ok := got.(*tensor.Dense)
	require.True(t, ok, "expected *tensor.Dense, got %T", got)
	require.True(t, d.IsScalar(), "expected 0-d Dense, got shape %v", d.Shape())
	return d.ScalarValue()
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
			assert.Equal(t, tc.expected, asDenseScalar(t, got))
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
			assert.Equal(t, tc.expected, asDenseScalar(t, got))
		})
	}
}

func TestScalarBinOp_DenseAdd_Float32(t *testing.T) {
	a := denseScalar(t, tensor.Float32, []float32{1.5})
	b := denseScalar(t, tensor.Float32, []float32{2.5})

	got := runScalarBinOp(t, addOpType, false, a, b)
	assert.Equal(t, float32(4.0), asDenseScalar(t, got))
}

func TestScalarBinOp_DenseAdd_Int32(t *testing.T) {
	a := denseScalar(t, tensor.Int32, []int32{3})
	b := denseScalar(t, tensor.Int32, []int32{4})

	got := runScalarBinOp(t, addOpType, false, a, b)
	assert.Equal(t, int32(7), asDenseScalar(t, got))
}

// TestScalarBinOp_DenseLt_RetSame covers the retSame=true path for a
// comparison op fed 0-d *tensor.Dense operands: the boolean result must be
// converted back to the operand dtype (1.0/0.0) and, since the inputs were
// Dense, re-boxed as a 0-d Dense rather than left as a boxed *F32.
func TestScalarBinOp_DenseLt_RetSame(t *testing.T) {
	a := denseScalar(t, tensor.Float32, []float32{1})
	b := denseScalar(t, tensor.Float32, []float32{2})

	got := runScalarBinOp(t, ltOpType, true, a, b)
	assert.Equal(t, float32(1.0), asDenseScalar(t, got))

	a = denseScalar(t, tensor.Float32, []float32{2})
	b = denseScalar(t, tensor.Float32, []float32{1})

	got = runScalarBinOp(t, ltOpType, true, a, b)
	assert.Equal(t, float32(0.0), asDenseScalar(t, got))
}

// TestScalarBinOp_DenseUint32_CleanError covers a 0-d Dense of a dtype
// scalarBinOp.Do's type switch doesn't handle (uint32 is not among
// float64/float32/int/int32/int64/byte/bool). denseToScalar must not
// attempt to box it (anyToScalar would panic on an unhandled type); it
// should fall through unnormalized so the existing "Unhandled Scalar Type"
// error path fires cleanly, exactly as it did before any Dense normalization
// existed - not a panic escaping the VM.
func TestScalarBinOp_DenseUint32_CleanError(t *testing.T) {
	a := denseScalar(t, tensor.Uint32, []uint32{1})
	b := denseScalar(t, tensor.Uint32, []uint32{2})

	g := NewGraph()
	na := NodeFromAny(g, a, WithName("a"))
	nb := NodeFromAny(g, b, WithName("b"))

	_, err := Lt(na, nb, false)
	require.NoError(t, err)

	m := NewTapeMachine(g)
	defer m.Close()

	assert.NotPanics(t, func() {
		err = m.RunAll()
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Unhandled Scalar Type")
}

// TestScalarBinOp_BoxedInBoxedOut confirms pure-gorgonia callers (both
// operands already boxed Scalars, e.g. *F32/*B - never *tensor.Dense) see
// zero behavior change: the result stays a boxed Scalar, not a Dense.
func TestScalarBinOp_BoxedInBoxedOut(t *testing.T) {
	g := NewGraph()
	na := NodeFromAny(g, NewF32(1), WithName("a"))
	nb := NodeFromAny(g, NewF32(2), WithName("b"))

	ret, err := Lt(na, nb, false)
	require.NoError(t, err)

	m := NewTapeMachine(g)
	defer m.Close()
	require.NoError(t, m.RunAll())

	bv, ok := ret.Value().(*B)
	require.True(t, ok, "expected *B (boxed Scalar), got %T", ret.Value())
	assert.True(t, bool(*bv))
}
