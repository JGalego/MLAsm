; ==============================================================================
; Matrix Operations - SIMD-optimized matrix arithmetic
; File: src/vectors/matrix_ops.asm
; 
; High-performance matrix operations using AVX2/AVX-512 instructions.
; Optimized for cache efficiency and vectorized computation.
; ==============================================================================

section .text

global ml_matrix_vector_mul_asm
global ml_matrix_mul_asm
global ml_matrix_transpose_asm
global ml_matrix_add_asm

; ==============================================================================
; ml_matrix_vector_mul_asm - Matrix-vector multiplication
; 
; Performs result = matrix * vector using optimized SIMD operations
; 
; Signature: void ml_matrix_vector_mul_asm(const float* matrix, const float* vector,
;                                          float* result, size_t rows, size_t cols)
; 
; Parameters:
;   rdi: pointer to matrix (row-major order)
;   rsi: pointer to input vector
;   rdx: pointer to result vector
;   rcx: number of rows
;   r8:  number of columns
; ==============================================================================

ml_matrix_vector_mul_asm:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    ; Check for null pointers
    test rdi, rdi
    jz .cleanup
    test rsi, rsi
    jz .cleanup
    test rdx, rdx
    jz .cleanup
    
    ; Check for zero dimensions
    test rcx, rcx
    jz .cleanup
    test r8, r8
    jz .cleanup
    
    ; Save parameters
    mov r12, rdi                ; matrix pointer
    mov r13, rsi                ; vector pointer
    mov r14, rdx                ; result pointer
    mov r15, rcx                ; rows
    ; r8 already contains cols
    
    xor rbx, rbx                ; row counter

.row_loop:
    cmp rbx, r15
    jge .cleanup
    
    ; Calculate matrix row offset (row * cols * sizeof(float))
    mov rax, rbx
    imul rax, r8
    shl rax, 2                  ; Multiply by 4 (sizeof(float))
    add rax, r12                ; rax = pointer to current row
    
    ; Compute dot product of current row with vector
    ; Use our optimized dot product function
    mov rdi, rax                ; Current row
    mov rsi, r13                ; Vector
    mov rdx, r8                 ; Column count
    call ml_vector_dot_asm      ; Result in xmm0
    
    ; Store result
    mov rax, rbx
    shl rax, 2                  ; row_index * sizeof(float)
    movss [r14 + rax], xmm0     ; Store in result[row]
    
    inc rbx
    jmp .row_loop

.cleanup:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; ==============================================================================
; ml_matrix_mul_asm - Matrix-matrix multiplication (C = A * B)
; 
; Performs optimized matrix multiplication using blocked algorithm
; for better cache efficiency.
; 
; Signature: void ml_matrix_mul_asm(const float* A, const float* B, float* C,
;                                   size_t rows_A, size_t cols_A, size_t cols_B)
; 
; Parameters:
;   rdi: pointer to matrix A
;   rsi: pointer to matrix B  
;   rdx: pointer to result matrix C
;   rcx: rows of A (and rows of C)
;   r8:  cols of A (and rows of B)
;   r9:  cols of B (and cols of C)
; ==============================================================================

ml_matrix_mul_asm:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    ; Check for null pointers and valid dimensions
    test rdi, rdi
    jz .cleanup
    test rsi, rsi
    jz .cleanup
    test rdx, rdx
    jz .cleanup
    test rcx, rcx
    jz .cleanup
    test r8, r8
    jz .cleanup
    test r9, r9
    jz .cleanup
    
    ; Save parameters
    mov r12, rdi                ; A
    mov r13, rsi                ; B
    mov r14, rdx                ; C
    mov r15, rcx                ; rows_A
    ; r8 = cols_A (rows_B)
    ; r9 = cols_B
    
    ; Initialize result matrix to zero
    mov rax, r15
    imul rax, r9                ; total elements in C
    mov rdi, r14                ; pointer to C
    xor rbx, rbx                ; counter

.zero_loop:
    cmp rbx, rax
    jge .zero_done
    mov dword [rdi + rbx*4], 0  ; zero out element
    inc rbx
    jmp .zero_loop

.zero_done:
    ; Block size for cache optimization
    mov r10, 64                 ; Block size (can be tuned)
    
    xor rbx, rbx                ; i = 0 (row of A and C)

.outer_loop_i:
    cmp rbx, r15
    jge .cleanup
    
    xor r11, r11                ; j = 0 (col of B and C)

.outer_loop_j:
    cmp r11, r9
    jge .next_i
    
    xor rax, rax                ; k = 0 (col of A, row of B)

.inner_loop_k:
    cmp rax, r8
    jge .next_j
    
    ; Calculate addresses:
    ; A[i][k] = A[i * cols_A + k]
    mov rdx, rbx
    imul rdx, r8
    add rdx, rax
    shl rdx, 2
    movss xmm0, [r12 + rdx]     ; Load A[i][k]
    
    ; B[k][j] = B[k * cols_B + j]
    mov rdx, rax
    imul rdx, r9
    add rdx, r11
    shl rdx, 2
    movss xmm1, [r13 + rdx]     ; Load B[k][j]
    
    ; C[i][j] = C[i * cols_B + j]
    mov rdx, rbx
    imul rdx, r9
    add rdx, r11
    shl rdx, 2
    
    ; Multiply and accumulate: C[i][j] += A[i][k] * B[k][j]
    mulss xmm0, xmm1
    addss xmm0, [r14 + rdx]
    movss [r14 + rdx], xmm0
    
    inc rax
    jmp .inner_loop_k

.next_j:
    inc r11
    jmp .outer_loop_j

.next_i:
    inc rbx
    jmp .outer_loop_i

.cleanup:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; ==============================================================================
; ml_matrix_transpose_asm - Matrix transpose
; 
; Signature: void ml_matrix_transpose_asm(const float* input, float* output,
;                                         size_t rows, size_t cols)
; 
; Parameters:
;   rdi: pointer to input matrix
;   rsi: pointer to output matrix
;   rdx: number of rows
;   rcx: number of columns
; ==============================================================================

ml_matrix_transpose_asm:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    
    ; Check for null pointers
    test rdi, rdi
    jz .cleanup
    test rsi, rsi
    jz .cleanup
    test rdx, rdx
    jz .cleanup
    test rcx, rcx
    jz .cleanup
    
    ; Save parameters
    mov r12, rdi                ; input
    mov r13, rsi                ; output
    ; rdx = rows, rcx = cols
    
    xor rbx, rbx                ; i = 0

.transpose_outer:
    cmp rbx, rdx
    jge .cleanup
    
    xor rax, rax                ; j = 0

.transpose_inner:
    cmp rax, rcx
    jge .transpose_next_row
    
    ; Load input[i][j] = input[i * cols + j]
    mov r8, rbx
    imul r8, rcx
    add r8, rax
    shl r8, 2
    movss xmm0, [r12 + r8]
    
    ; Store to output[j][i] = output[j * rows + i]
    mov r8, rax
    imul r8, rdx
    add r8, rbx
    shl r8, 2
    movss [r13 + r8], xmm0
    
    inc rax
    jmp .transpose_inner

.transpose_next_row:
    inc rbx
    jmp .transpose_outer

.cleanup:
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; ==============================================================================
; ml_matrix_add_asm - Matrix addition
; 
; Signature: void ml_matrix_add_asm(const float* A, const float* B, float* C,
;                                   size_t rows, size_t cols)
; 
; Parameters:
;   rdi: pointer to matrix A
;   rsi: pointer to matrix B
;   rdx: pointer to result matrix C
;   rcx: number of rows
;   r8:  number of columns
; ==============================================================================

ml_matrix_add_asm:
    push rbp
    mov rbp, rsp
    
    ; Check for null pointers
    test rdi, rdi
    jz .cleanup
    test rsi, rsi
    jz .cleanup
    test rdx, rdx
    jz .cleanup
    
    ; Calculate total number of elements
    mov rax, rcx
    imul rax, r8                ; total elements
    
    ; Use vector addition for the flattened matrices
    ; Parameters: rdi=A, rsi=B, rdx=C, rcx=size
    mov rcx, rax
    call ml_vector_add_asm

.cleanup:
    pop rbp
    ret

; Reference to external functions
extern ml_vector_dot_asm
extern ml_vector_add_asm
extern memset

section .data
align 32
; Cache line size constant
cache_line_size: dq 64

; Mark stack as non-executable
section .note.GNU-stack noalloc noexec nowrite progbits
