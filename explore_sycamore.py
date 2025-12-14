
import cirq
import cirq_google
from cirq_google import SycamoreTargetGateset

def explore():
    target_gateset = SycamoreTargetGateset()
    print("Methods of SycamoreTargetGateset:")
    for method in dir(target_gateset):
        if not method.startswith('_'):
            print(method)

    print("\nAttempting to decompose a CNOT operation:")
    q0, q1 = cirq.LineQubit.range(2)
    cnot_op = cirq.CNOT(q0, q1)
    
    # Try using the decompose_operation method if it exists, or just cirq.decompose
    try:
        decomposed = target_gateset.decompose_to_target_gateset(cnot_op, 1) # Argument 1 is 'moment_index' often ignored
        print("Decomposed using decompose_to_target_gateset (if available):")
        print(decomposed)
    except AttributeError:
        print("decompose_to_target_gateset not found.")

    # Check how optimize_for_target_gateset works
    # It usually iterates optimizers.
    
    # Let's see if we can just use cirq.decompose with an interceptor or similar.
    # Actually, SycamoreTargetGateset likely has a method to decompose a single operation.
    
    # Let's inspect the decompose method
    try:
        # Assuming it might be called 'decompose' or similar
        print("\nTrying cirq.decompose with keep constraint:")
        def keep(op):
            return op in target_gateset
            
        decomposed = cirq.decompose(cnot_op, keep=keep)
        print(f"Result of cirq.decompose: {len(decomposed)} ops")
        for op in decomposed:
            print(op)
    except Exception as e:
        print(f"Error in cirq.decompose: {e}")

if __name__ == "__main__":
    explore()
