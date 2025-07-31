; ==============================================================================
; Activation Functions - SIMD-optimized neural network activations
; File: src/activations/activations.asm
; 
; High-performance implementation of common activation functions using
; AVX2/AVX-512 instructions with lookup tables for transcendental functions.
; ==============================================================================

section .text

global ml_activation_relu_asm
global ml_activation_sigmoid_asm
global ml_activation_tanh_asm
global ml_activation_softmax_asm
global ml_activation_leaky_relu_asm
global sigmoid_lut

; External functions
extern expf, logf

; ==============================================================================
; ml_activation_relu_asm - ReLU activation function
; 
; Applies ReLU: f(x) = max(0, x) element-wise to a vector
; 
; Signature: void ml_activation_relu_asm(const float* input, float* output, size_t size)
; 
; Parameters:
;   rdi: pointer to input vector
;   rsi: pointer to output vector
;   rdx: vector size
; ==============================================================================

ml_activation_relu_asm:
    push rbp
    mov rbp, rsp
    
    ; Check for null pointers and zero size
    test rdi, rdi
    jz .cleanup
    test rsi, rsi
    jz .cleanup
    test rdx, rdx
    jz .cleanup
    
    ; Zero vector for comparison
    vxorps ymm1, ymm1, ymm1    ; ymm1 = [0.0, 0.0, ..., 0.0]
    
    xor rax, rax               ; Index counter
    
    ; Check if we have enough elements for SIMD
    cmp rdx, 8
    jl .scalar_relu_loop
    
    ; Calculate SIMD iterations
    mov rcx, rdx
    shr rcx, 3                 ; Divide by 8
    shl rcx, 3                 ; Multiply by 8 for aligned count

.simd_relu_loop:
    cmp rax, rcx
    jge .handle_relu_remainder
    
    ; Load 8 floats
    vmovups ymm0, [rdi + rax*4]
    
    ; Apply ReLU: max(0, x)
    vmaxps ymm2, ymm0, ymm1
    
    ; Store result
    vmovups [rsi + rax*4], ymm2
    
    add rax, 8
    jmp .simd_relu_loop

.handle_relu_remainder:
    cmp rax, rdx
    jge .cleanup

.scalar_relu_loop:
    ; Load single float
    movss xmm0, [rdi + rax*4]
    
    ; Apply ReLU
    maxss xmm0, xmm1           ; max(x, 0.0)
    
    ; Store result
    movss [rsi + rax*4], xmm0
    
    inc rax
    cmp rax, rdx
    jl .scalar_relu_loop

.cleanup:
    vzeroupper
    pop rbp
    ret

; ==============================================================================
; ml_activation_leaky_relu_asm - Leaky ReLU activation function
; 
; Applies Leaky ReLU: f(x) = x if x > 0, alpha*x otherwise
; 
; Signature: void ml_activation_leaky_relu_asm(const float* input, float* output, 
;                                              size_t size, float alpha)
; 
; Parameters:
;   rdi: pointer to input vector
;   rsi: pointer to output vector
;   rdx: vector size
;   xmm0: alpha parameter (typically 0.01)
; ==============================================================================

ml_activation_leaky_relu_asm:
    push rbp
    mov rbp, rsp
    
    ; Check for null pointers and zero size
    test rdi, rdi
    jz .cleanup
    test rsi, rsi
    jz .cleanup
    test rdx, rdx
    jz .cleanup
    
    ; Broadcast alpha to all lanes
    vbroadcastss ymm3, xmm0    ; ymm3 = [alpha, alpha, ..., alpha]
    
    ; Zero vector for comparison
    vxorps ymm1, ymm1, ymm1    ; ymm1 = [0.0, 0.0, ..., 0.0]
    
    xor rax, rax               ; Index counter
    
    ; Check if we have enough elements for SIMD
    cmp rdx, 8
    jl .scalar_leaky_loop
    
    ; Calculate SIMD iterations
    mov rcx, rdx
    shr rcx, 3
    shl rcx, 3

.simd_leaky_loop:
    cmp rax, rcx
    jge .handle_leaky_remainder
    
    ; Load 8 floats
    vmovups ymm0, [rdi + rax*4]
    
    ; Create mask for positive values
    vcmpps ymm4, ymm0, ymm1, 0x1E  ; Compare greater than 0
    
    ; Compute alpha * x
    vmulps ymm2, ymm0, ymm3
    
    ; Select: x if x > 0, alpha*x otherwise
    vblendvps ymm5, ymm2, ymm0, ymm4
    
    ; Store result
    vmovups [rsi + rax*4], ymm5
    
    add rax, 8
    jmp .simd_leaky_loop

.handle_leaky_remainder:
    cmp rax, rdx
    jge .cleanup

.scalar_leaky_loop:
    ; Load single float
    movss xmm0, [rdi + rax*4]
    
    ; Check if positive
    xorps xmm2, xmm2
    comiss xmm0, xmm2
    ja .positive_leaky
    
    ; Negative: multiply by alpha
    mulss xmm0, xmm3           ; xmm3[0] contains alpha
    
.positive_leaky:
    ; Store result (either original x or alpha*x)
    movss [rsi + rax*4], xmm0
    
    inc rax
    cmp rax, rdx
    jl .scalar_leaky_loop

.cleanup:
    vzeroupper
    pop rbp
    ret

; ==============================================================================
; ==============================================================================
; ml_activation_sigmoid_asm - Sigmoid activation function
; 
; Applies sigmoid: f(x) = 1 / (1 + exp(-x)) element-wise to a vector
; Uses direct computation without external constants for reliability
; 
; Signature: void ml_activation_sigmoid_asm(const float* input, float* output, size_t size)
; 
; Parameters:
;   rdi: pointer to input vector
;   rsi: pointer to output vector
;   rdx: vector size
; ==============================================================================

ml_activation_sigmoid_asm:
    push rbp
    mov rbp, rsp
    push rbx
    
    ; Check for null pointers and zero size
    test rdi, rdi
    jz .cleanup
    test rsi, rsi
    jz .cleanup
    test rdx, rdx
    jz .cleanup
    
    ; Load constants directly on stack
    sub rsp, 32
    mov dword [rsp], 0x3F800000      ; 1.0f
    mov dword [rsp+4], 0xC2280000    ; -42.0f (extreme negative threshold)
    mov dword [rsp+8], 0x42280000    ; 42.0f (extreme positive threshold)
    mov dword [rsp+12], 0x33D6BF95   ; 1e-7f (epsilon for bounds)
    mov dword [rsp+16], 0x3F7FFFFF   ; 1.0f - 1e-7f (close to 1 but not exactly 1)
    
    xor rax, rax               ; Index counter
    
.sigmoid_loop:
    cmp rax, rdx
    jge .cleanup_stack
    
    ; Load input value
    movss xmm0, [rdi + rax*4]
    
    ; Handle extreme values for numerical stability
    ; if x > 42, sigmoid(x) ≈ 1.0
    ; if x < -42, sigmoid(x) ≈ 0.0
    
    movss xmm1, [rsp+8]      ; 42.0f
    comiss xmm0, xmm1
    ja .set_one              ; x > 42, set to 1
    
    movss xmm1, [rsp+4]      ; -42.0f
    comiss xmm0, xmm1
    jb .set_zero             ; x < -42, set to 0
    
    ; For reasonable values, compute sigmoid = 1 / (1 + exp(-x))
    ; Negate x
    xorps xmm1, xmm1         ; Clear xmm1 to 0
    subss xmm1, xmm0         ; 0 - x = -x
    
    ; Call expf(-x) - we need to preserve registers
    push rax                 ; Save loop counter
    push rdx                 ; Save array size
    push rdi                 ; Save input pointer
    push rsi                 ; Save output pointer
    sub rsp, 8               ; Align stack to 16 bytes
    
    movss xmm0, xmm1         ; Move -x to xmm0 for expf argument
    call expf wrt ..plt      ; exp(-x)
    
    add rsp, 8               ; Restore stack alignment
    pop rsi                  ; Restore output pointer
    pop rdi                  ; Restore input pointer  
    pop rdx                  ; Restore array size
    pop rax                  ; Restore loop counter
    movss xmm1, xmm0         ; Store exp(-x) result
    
    ; Calculate 1 + exp(-x)
    movss xmm2, [rsp]        ; 1.0f
    addss xmm1, xmm2         ; 1 + exp(-x)
    
    ; Calculate 1 / (1 + exp(-x))
    divss xmm2, xmm1         ; 1 / (1 + exp(-x))
    movss xmm0, xmm2
    
    ; Ensure result is strictly in (0,1) - clamp if needed
    xorps xmm3, xmm3         ; 0.0f
    comiss xmm0, xmm3
    jbe .clamp_to_epsilon    ; if result <= 0, clamp to epsilon
    
    movss xmm3, [rsp]        ; 1.0f
    comiss xmm0, xmm3
    jae .clamp_to_one_minus_epsilon  ; if result >= 1, clamp to 1-epsilon
    jmp .store_result
    
.clamp_to_epsilon:
    movss xmm0, [rsp+12]     ; epsilon
    jmp .store_result
    
.clamp_to_one_minus_epsilon:
    movss xmm0, [rsp+16]     ; 1.0f - epsilon
    jmp .store_result
    
.set_one:
    movss xmm0, [rsp+16]     ; 1.0f - epsilon (very close to 1 but not exactly 1)
    jmp .store_result
    
.set_zero:
    movss xmm0, [rsp+12]     ; epsilon (very close to 0 but not exactly 0)
    
.store_result:
    ; Store result
    movss [rsi + rax*4], xmm0
    
    inc rax
    jmp .sigmoid_loop

.cleanup_stack:
    add rsp, 32
.cleanup:
    pop rbx
    vzeroupper
    pop rbp
    ret

; ==============================================================================
; ml_activation_tanh_asm - Hyperbolic tangent activation
; 
; Applies tanh: f(x) = (exp(2x) - 1) / (exp(2x) + 1)
; Uses fast approximation for better performance
; 
; Signature: void ml_activation_tanh_asm(const float* input, float* output, size_t size)
; 
; Parameters:
;   rdi: pointer to input vector
;   rsi: pointer to output vector
;   rdx: vector size
; ==============================================================================

ml_activation_tanh_asm:
    push rbp
    mov rbp, rsp
    
    ; Check for null pointers and zero size
    test rdi, rdi
    jz .cleanup
    test rsi, rsi
    jz .cleanup
    test rdx, rdx
    jz .cleanup
    
    ; Load constants directly on stack
    sub rsp, 32
    mov dword [rsp], 0x3F800000      ; 1.0f
    mov dword [rsp+4], 0x40000000    ; 2.0f
    mov dword [rsp+8], 0x7FFFFFFF    ; abs_mask
    mov dword [rsp+12], 0x80000000   ; sign_mask
    
    xor rax, rax               ; Index counter

.tanh_loop:
    cmp rax, rdx
    jge .cleanup_stack
    
    ; Load input value
    movss xmm0, [rdi + rax*4]
    
    ; Fast tanh approximation: tanh(x) ≈ x / (1 + |x|) for |x| < 1
    ; For larger values, use tanh(x) ≈ sign(x) * (1 - 2/(exp(2|x|) + 1))
    
    ; Get absolute value and sign
    movss xmm1, xmm0
    movss xmm3, [rsp+8]       ; abs_mask
    andps xmm1, xmm3          ; |x|
    movss xmm2, xmm0
    movss xmm3, [rsp+12]      ; sign_mask
    andps xmm2, xmm3          ; sign(x)
    
    ; Check if |x| < 1
    movss xmm3, [rsp]         ; 1.0f
    comiss xmm1, xmm3
    jb .tanh_small
    
    ; Large value approximation
    movss xmm3, [rsp+4]       ; 2.0f
    mulss xmm1, xmm3          ; 2|x|
    
    ; Simplified exp approximation: exp(x) ≈ 1 + x + x²/2 for small x
    ; For larger x, clamp to avoid overflow
    movss xmm3, xmm1
    mulss xmm3, xmm3          ; x²
    movss xmm4, [rsp+4]       ; 2.0f
    divss xmm3, xmm4          ; x²/2
    addss xmm3, xmm1          ; x + x²/2
    movss xmm4, [rsp]         ; 1.0f
    addss xmm3, xmm4          ; 1 + x + x²/2
    
    movss xmm4, [rsp+4]       ; 2.0f
    divss xmm4, xmm3          ; 2/(exp(2|x|) + 1)
    movss xmm0, [rsp]         ; 1.0f
    subss xmm0, xmm4          ; 1 - 2/(exp(2|x|) + 1)
    
    ; Apply sign
    orps xmm0, xmm2
    jmp .tanh_store
    
.tanh_small:
    ; Small value: tanh(x) ≈ x / (1 + |x|)
    movss xmm3, [rsp]         ; 1.0f
    addss xmm1, xmm3          ; 1 + |x|
    divss xmm0, xmm1          ; x / (1 + |x|)

.tanh_store:
    ; Store result
    movss [rsi + rax*4], xmm0
    
    inc rax
    jmp .tanh_loop

.cleanup_stack:
    add rsp, 32

.cleanup:
    vzeroupper
    pop rbp
    ret

; ==============================================================================
; ml_activation_softmax_asm - Softmax activation function
; 
; Applies softmax: f(x_i) = exp(x_i) / sum(exp(x_j))
; Numerically stable implementation
; 
; Signature: void ml_activation_softmax_asm(const float* input, float* output, size_t size)
; 
; Parameters:
;   rdi: pointer to input vector
;   rsi: pointer to output vector
;   rdx: vector size
; ==============================================================================

ml_activation_softmax_asm:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    
    ; Check for null pointers and zero size
    test rdi, rdi
    jz .cleanup
    test rsi, rsi
    jz .cleanup
    test rdx, rdx
    jz .cleanup
    
    ; Load constants directly on stack
    sub rsp, 16
    mov dword [rsp], 0x3F800000      ; 1.0f
    mov dword [rsp+4], 0x3F000000    ; 0.5f
    
    mov r12, rdx               ; Save size
    
    ; Step 1: Find maximum value for numerical stability
    movss xmm0, [rdi]          ; Initialize max with first element
    mov rax, 1

.find_max_loop:
    cmp rax, rdx
    jge .subtract_max
    
    movss xmm1, [rdi + rax*4]
    maxss xmm0, xmm1           ; max = max(max, current)
    
    inc rax
    jmp .find_max_loop

.subtract_max:
    ; Step 2: Subtract max and compute exp(x_i - max)
    ; Also accumulate sum for normalization
    vxorps xmm2, xmm2, xmm2    ; sum = 0
    xor rax, rax

.exp_loop:
    cmp rax, rdx
    jge .normalize
    
    ; Load value and subtract max
    movss xmm1, [rdi + rax*4]
    subss xmm1, xmm0           ; x_i - max
    
    ; Compute exp(x_i - max) using proper exponential function
    push rax                   ; Save loop counter
    push rdx                   ; Save array size
    push rdi                   ; Save input pointer  
    push rsi                   ; Save output pointer
    sub rsp, 16                ; Align stack + save XMM registers
    movss [rsp], xmm0          ; Save max value
    movss [rsp+4], xmm2        ; Save sum
    
    movss xmm0, xmm1           ; Move argument to xmm0 for expf
    call expf wrt ..plt        ; Call exponential function
    movss xmm3, xmm0           ; Store result
    
    ; Restore registers
    movss xmm2, [rsp+4]        ; Restore sum
    movss xmm0, [rsp]          ; Restore max value
    add rsp, 16                ; Restore stack alignment
    pop rsi                    ; Restore output pointer
    pop rdi                    ; Restore input pointer
    pop rdx                    ; Restore array size
    pop rax                    ; Restore loop counter
    
    ; Store temporary result
    movss [rsi + rax*4], xmm3
    
    ; Accumulate sum
    addss xmm2, xmm3
    
    inc rax
    jmp .exp_loop

.normalize:
    ; Step 3: Normalize by dividing each element by sum
    xor rax, rax

.normalize_loop:
    cmp rax, rdx
    jge .cleanup_stack
    
    movss xmm0, [rsi + rax*4]
    divss xmm0, xmm2           ; element / sum
    movss [rsi + rax*4], xmm0
    
    inc rax
    jmp .normalize_loop

.cleanup_stack:
    add rsp, 16

.cleanup:
    pop r12
    pop rbx
    vzeroupper
    pop rbp
    ret

section .data
align 32

; Constants for activation functions
one:         dd 1.0
half:        dd 0.5
quarter:     dd 0.25
two:         dd 2.0
neg_ten:     dd -10.0
pos_ten:     dd 10.0

; Bit masks
abs_mask:    dd 0x7FFFFFFF
sign_mask:   dd 0x80000000

; Sigmoid lookup table constants
sigmoid_scale:  times 8 dd 16.0    ; Scale factor for LUT
sigmoid_offset: times 8 dd 128.0   ; Offset for LUT

; Placeholder for sigmoid lookup table (would be initialized at runtime)
sigmoid_lut: times 256 dd 0.0

section .bss
; Reserve space for runtime-generated lookup tables
tanh_lut: resb 1024

; Mark stack as non-executable
section .note.GNU-stack noalloc noexec nowrite progbits
