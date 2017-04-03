(* mips - generated by L3 - Thu Mar 30 14:42:27 2017 *)

signature mips =
sig

structure Map : Map

(* -------------------------------------------------------------------------
   Types
   ------------------------------------------------------------------------- *)

type Index = { Index: BitsN.nbit, P: bool, index'rst: BitsN.nbit }

type Random = { Random: BitsN.nbit, random'rst: BitsN.nbit }

type Wired = { Wired: BitsN.nbit, wired'rst: BitsN.nbit }

type EntryLo =
  { C: BitsN.nbit, D: bool, G: bool, PFN: BitsN.nbit, V: bool,
    entrylo'rst: BitsN.nbit }

type PageMask = { Mask: BitsN.nbit, pagemask'rst: BitsN.nbit }

type EntryHi =
  { ASID: BitsN.nbit, R: BitsN.nbit, VPN2: BitsN.nbit,
    entryhi'rst: BitsN.nbit }

type StatusRegister =
  { BEV: bool, CU0: bool, ERL: bool, EXL: bool, FR: bool, IE: bool,
    IM: BitsN.nbit, KSU: BitsN.nbit, KX: bool, RE: bool, SX: bool,
    UX: bool, statusregister'rst: BitsN.nbit }

type ConfigRegister =
  { AR: BitsN.nbit, AT: BitsN.nbit, BE: bool, K0: BitsN.nbit, M: bool,
    MT: BitsN.nbit, configregister'rst: BitsN.nbit }

type ConfigRegister1 =
  { C2: bool, CA: bool, DA: BitsN.nbit, DL: BitsN.nbit, DS: BitsN.nbit,
    EP: bool, FP: bool, IA: BitsN.nbit, IL: BitsN.nbit, IS: BitsN.nbit,
    M: bool, MD: bool, MMUSize: BitsN.nbit, PC: bool, WR: bool }

type ConfigRegister2 =
  { M: bool, SA: BitsN.nbit, SL: BitsN.nbit, SS: BitsN.nbit,
    SU: BitsN.nbit, TA: BitsN.nbit, TL: BitsN.nbit, TS: BitsN.nbit,
    TU: BitsN.nbit }

type ConfigRegister3 =
  { DSPP: bool, LPA: bool, M: bool, MT: bool, SM: bool, SP: bool,
    TL: bool, ULRI: bool, VEIC: bool, VInt: bool,
    configregister3'rst: BitsN.nbit }

type ConfigRegister6 =
  { LTLB: bool, TLBSize: BitsN.nbit, configregister6'rst: BitsN.nbit }

type CauseRegister =
  { BD: bool, ExcCode: BitsN.nbit, IP: BitsN.nbit, TI: bool,
    causeregister'rst: BitsN.nbit }

type Context =
  { BadVPN2: BitsN.nbit, PTEBase: BitsN.nbit, context'rst: BitsN.nbit }

type XContext =
  { BadVPN2: BitsN.nbit, PTEBase: BitsN.nbit, R: BitsN.nbit,
    xcontext'rst: BitsN.nbit }

type HWREna =
  { CC: bool, CCRes: bool, CPUNum: bool, UL: bool, hwrena'rst: BitsN.nbit
    }

type CP0 =
  { BadVAddr: BitsN.nbit, Cause: CauseRegister, Compare: BitsN.nbit,
    Config: ConfigRegister, Config1: ConfigRegister1,
    Config2: ConfigRegister2, Config3: ConfigRegister3,
    Config6: ConfigRegister6, Context: Context, Count: BitsN.nbit,
    Debug: BitsN.nbit, EPC: BitsN.nbit, EntryHi: EntryHi,
    EntryLo0: EntryLo, EntryLo1: EntryLo, ErrCtl: BitsN.nbit,
    ErrorEPC: BitsN.nbit, HWREna: HWREna, Index: Index,
    LLAddr: BitsN.nbit, PRId: BitsN.nbit, PageMask: PageMask,
    Random: Random, Status: StatusRegister, UsrLocal: BitsN.nbit,
    Wired: Wired, XContext: XContext }

datatype ExceptionType
  = Int | Mod | TLBL | TLBS | AdEL | AdES | Sys | Bp | ResI | CpU | Ov
  | Tr | XTLBRefillL | XTLBRefillS

datatype IorD = INSTRUCTION | DATA

datatype LorS = LOAD | STORE

datatype Branch
  = BEQ of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | BEQL of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | BGEZ of BitsN.nbit * BitsN.nbit
  | BGEZAL of BitsN.nbit * BitsN.nbit
  | BGEZALL of BitsN.nbit * BitsN.nbit
  | BGEZL of BitsN.nbit * BitsN.nbit
  | BGTZ of BitsN.nbit * BitsN.nbit
  | BGTZL of BitsN.nbit * BitsN.nbit
  | BLEZ of BitsN.nbit * BitsN.nbit
  | BLEZL of BitsN.nbit * BitsN.nbit
  | BLTZ of BitsN.nbit * BitsN.nbit
  | BLTZAL of BitsN.nbit * BitsN.nbit
  | BLTZALL of BitsN.nbit * BitsN.nbit
  | BLTZL of BitsN.nbit * BitsN.nbit
  | BNE of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | BNEL of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | J of BitsN.nbit
  | JAL of BitsN.nbit
  | JALR of BitsN.nbit * BitsN.nbit
  | JR of BitsN.nbit

datatype CP
  = DMFC0 of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | DMTC0 of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | MFC0 of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | MTC0 of BitsN.nbit * (BitsN.nbit * BitsN.nbit)

datatype Store
  = SB of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SC of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SCD of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SD of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SDL of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SDR of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SH of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SW of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SWL of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SWR of BitsN.nbit * (BitsN.nbit * BitsN.nbit)

datatype Load
  = LB of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | LBU of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | LD of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | LDL of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | LDR of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | LH of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | LHU of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | LL of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | LLD of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | LW of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | LWL of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | LWR of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | LWU of BitsN.nbit * (BitsN.nbit * BitsN.nbit)

datatype Trap
  = TEQ of BitsN.nbit * BitsN.nbit
  | TEQI of BitsN.nbit * BitsN.nbit
  | TGE of BitsN.nbit * BitsN.nbit
  | TGEI of BitsN.nbit * BitsN.nbit
  | TGEIU of BitsN.nbit * BitsN.nbit
  | TGEU of BitsN.nbit * BitsN.nbit
  | TLT of BitsN.nbit * BitsN.nbit
  | TLTI of BitsN.nbit * BitsN.nbit
  | TLTIU of BitsN.nbit * BitsN.nbit
  | TLTU of BitsN.nbit * BitsN.nbit
  | TNE of BitsN.nbit * BitsN.nbit
  | TNEI of BitsN.nbit * BitsN.nbit

datatype Shift
  = DSLL of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | DSLL32 of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | DSLLV of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | DSRA of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | DSRA32 of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | DSRAV of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | DSRL of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | DSRL32 of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | DSRLV of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SLL of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SLLV of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SRA of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SRAV of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SRL of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SRLV of BitsN.nbit * (BitsN.nbit * BitsN.nbit)

datatype MultDiv
  = DDIV of BitsN.nbit * BitsN.nbit
  | DDIVU of BitsN.nbit * BitsN.nbit
  | DIV of BitsN.nbit * BitsN.nbit
  | DIVU of BitsN.nbit * BitsN.nbit
  | DMULT of BitsN.nbit * BitsN.nbit
  | DMULTU of BitsN.nbit * BitsN.nbit
  | MADD of BitsN.nbit * BitsN.nbit
  | MADDU of BitsN.nbit * BitsN.nbit
  | MFHI of BitsN.nbit
  | MFLO of BitsN.nbit
  | MSUB of BitsN.nbit * BitsN.nbit
  | MSUBU of BitsN.nbit * BitsN.nbit
  | MTHI of BitsN.nbit
  | MTLO of BitsN.nbit
  | MUL of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | MULT of BitsN.nbit * BitsN.nbit
  | MULTU of BitsN.nbit * BitsN.nbit

datatype ArithR
  = ADD of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | ADDU of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | AND of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | DADD of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | DADDU of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | DSUB of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | DSUBU of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | MOVN of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | MOVZ of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | NOR of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | OR of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SLT of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SLTU of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SUB of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SUBU of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | XOR of BitsN.nbit * (BitsN.nbit * BitsN.nbit)

datatype ArithI
  = ADDI of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | ADDIU of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | ANDI of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | DADDI of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | DADDIU of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | LUI of BitsN.nbit * BitsN.nbit
  | ORI of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SLTI of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | SLTIU of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | XORI of BitsN.nbit * (BitsN.nbit * BitsN.nbit)

datatype instruction
  = ArithI of ArithI
  | ArithR of ArithR
  | BREAK
  | Branch of Branch
  | CACHE of BitsN.nbit * (BitsN.nbit * BitsN.nbit)
  | CP of CP
  | ERET
  | Load of Load
  | MultDiv of MultDiv
  | RDHWR of BitsN.nbit * BitsN.nbit
  | ReservedInstruction
  | SYNC of BitsN.nbit
  | SYSCALL
  | Shift of Shift
  | Store of Store
  | TLBP
  | TLBR
  | TLBWI
  | TLBWR
  | Trap of Trap
  | Unpredictable
  | WAIT

datatype maybe_instruction
  = FAIL of string | OK of instruction | WORD32 of BitsN.nbit

(* -------------------------------------------------------------------------
   Exceptions
   ------------------------------------------------------------------------- *)

exception UNPREDICTABLE of string

(* -------------------------------------------------------------------------
   Functions
   ------------------------------------------------------------------------- *)

structure Cast:
sig

val natToExceptionType:Nat.nat -> ExceptionType
val ExceptionTypeToNat:ExceptionType-> Nat.nat
val stringToExceptionType:string -> ExceptionType
val ExceptionTypeToString:ExceptionType-> string
val natToIorD:Nat.nat -> IorD
val IorDToNat:IorD-> Nat.nat
val stringToIorD:string -> IorD
val IorDToString:IorD-> string
val natToLorS:Nat.nat -> LorS
val LorSToNat:LorS-> Nat.nat
val stringToLorS:string -> LorS
val LorSToString:LorS-> string

end

val BranchDelay: ((BitsN.nbit option) option) ref
val BranchTo: ((bool * BitsN.nbit) option) ref
val CP0: CP0 ref
val LLbit: (bool option) ref
val MEM: (BitsN.nbit Map.map) ref
val PC: BitsN.nbit ref
val exceptionSignalled: bool ref
val gpr: (BitsN.nbit Map.map) ref
val hi: (BitsN.nbit option) ref
val lo: (BitsN.nbit option) ref
val Index_Index_rupd: Index * BitsN.nbit -> Index
val Index_P_rupd: Index * bool -> Index
val Index_index'rst_rupd: Index * BitsN.nbit -> Index
val Random_Random_rupd: Random * BitsN.nbit -> Random
val Random_random'rst_rupd: Random * BitsN.nbit -> Random
val Wired_Wired_rupd: Wired * BitsN.nbit -> Wired
val Wired_wired'rst_rupd: Wired * BitsN.nbit -> Wired
val EntryLo_C_rupd: EntryLo * BitsN.nbit -> EntryLo
val EntryLo_D_rupd: EntryLo * bool -> EntryLo
val EntryLo_G_rupd: EntryLo * bool -> EntryLo
val EntryLo_PFN_rupd: EntryLo * BitsN.nbit -> EntryLo
val EntryLo_V_rupd: EntryLo * bool -> EntryLo
val EntryLo_entrylo'rst_rupd: EntryLo * BitsN.nbit -> EntryLo
val PageMask_Mask_rupd: PageMask * BitsN.nbit -> PageMask
val PageMask_pagemask'rst_rupd: PageMask * BitsN.nbit -> PageMask
val EntryHi_ASID_rupd: EntryHi * BitsN.nbit -> EntryHi
val EntryHi_R_rupd: EntryHi * BitsN.nbit -> EntryHi
val EntryHi_VPN2_rupd: EntryHi * BitsN.nbit -> EntryHi
val EntryHi_entryhi'rst_rupd: EntryHi * BitsN.nbit -> EntryHi
val StatusRegister_BEV_rupd: StatusRegister * bool -> StatusRegister
val StatusRegister_CU0_rupd: StatusRegister * bool -> StatusRegister
val StatusRegister_ERL_rupd: StatusRegister * bool -> StatusRegister
val StatusRegister_EXL_rupd: StatusRegister * bool -> StatusRegister
val StatusRegister_FR_rupd: StatusRegister * bool -> StatusRegister
val StatusRegister_IE_rupd: StatusRegister * bool -> StatusRegister
val StatusRegister_IM_rupd: StatusRegister * BitsN.nbit -> StatusRegister
val StatusRegister_KSU_rupd: StatusRegister * BitsN.nbit -> StatusRegister
val StatusRegister_KX_rupd: StatusRegister * bool -> StatusRegister
val StatusRegister_RE_rupd: StatusRegister * bool -> StatusRegister
val StatusRegister_SX_rupd: StatusRegister * bool -> StatusRegister
val StatusRegister_UX_rupd: StatusRegister * bool -> StatusRegister
val StatusRegister_statusregister'rst_rupd:
  StatusRegister * BitsN.nbit -> StatusRegister
val ConfigRegister_AR_rupd: ConfigRegister * BitsN.nbit -> ConfigRegister
val ConfigRegister_AT_rupd: ConfigRegister * BitsN.nbit -> ConfigRegister
val ConfigRegister_BE_rupd: ConfigRegister * bool -> ConfigRegister
val ConfigRegister_K0_rupd: ConfigRegister * BitsN.nbit -> ConfigRegister
val ConfigRegister_M_rupd: ConfigRegister * bool -> ConfigRegister
val ConfigRegister_MT_rupd: ConfigRegister * BitsN.nbit -> ConfigRegister
val ConfigRegister_configregister'rst_rupd:
  ConfigRegister * BitsN.nbit -> ConfigRegister
val ConfigRegister1_C2_rupd: ConfigRegister1 * bool -> ConfigRegister1
val ConfigRegister1_CA_rupd: ConfigRegister1 * bool -> ConfigRegister1
val ConfigRegister1_DA_rupd:
  ConfigRegister1 * BitsN.nbit -> ConfigRegister1
val ConfigRegister1_DL_rupd:
  ConfigRegister1 * BitsN.nbit -> ConfigRegister1
val ConfigRegister1_DS_rupd:
  ConfigRegister1 * BitsN.nbit -> ConfigRegister1
val ConfigRegister1_EP_rupd: ConfigRegister1 * bool -> ConfigRegister1
val ConfigRegister1_FP_rupd: ConfigRegister1 * bool -> ConfigRegister1
val ConfigRegister1_IA_rupd:
  ConfigRegister1 * BitsN.nbit -> ConfigRegister1
val ConfigRegister1_IL_rupd:
  ConfigRegister1 * BitsN.nbit -> ConfigRegister1
val ConfigRegister1_IS_rupd:
  ConfigRegister1 * BitsN.nbit -> ConfigRegister1
val ConfigRegister1_M_rupd: ConfigRegister1 * bool -> ConfigRegister1
val ConfigRegister1_MD_rupd: ConfigRegister1 * bool -> ConfigRegister1
val ConfigRegister1_MMUSize_rupd:
  ConfigRegister1 * BitsN.nbit -> ConfigRegister1
val ConfigRegister1_PC_rupd: ConfigRegister1 * bool -> ConfigRegister1
val ConfigRegister1_WR_rupd: ConfigRegister1 * bool -> ConfigRegister1
val ConfigRegister2_M_rupd: ConfigRegister2 * bool -> ConfigRegister2
val ConfigRegister2_SA_rupd:
  ConfigRegister2 * BitsN.nbit -> ConfigRegister2
val ConfigRegister2_SL_rupd:
  ConfigRegister2 * BitsN.nbit -> ConfigRegister2
val ConfigRegister2_SS_rupd:
  ConfigRegister2 * BitsN.nbit -> ConfigRegister2
val ConfigRegister2_SU_rupd:
  ConfigRegister2 * BitsN.nbit -> ConfigRegister2
val ConfigRegister2_TA_rupd:
  ConfigRegister2 * BitsN.nbit -> ConfigRegister2
val ConfigRegister2_TL_rupd:
  ConfigRegister2 * BitsN.nbit -> ConfigRegister2
val ConfigRegister2_TS_rupd:
  ConfigRegister2 * BitsN.nbit -> ConfigRegister2
val ConfigRegister2_TU_rupd:
  ConfigRegister2 * BitsN.nbit -> ConfigRegister2
val ConfigRegister3_DSPP_rupd: ConfigRegister3 * bool -> ConfigRegister3
val ConfigRegister3_LPA_rupd: ConfigRegister3 * bool -> ConfigRegister3
val ConfigRegister3_M_rupd: ConfigRegister3 * bool -> ConfigRegister3
val ConfigRegister3_MT_rupd: ConfigRegister3 * bool -> ConfigRegister3
val ConfigRegister3_SM_rupd: ConfigRegister3 * bool -> ConfigRegister3
val ConfigRegister3_SP_rupd: ConfigRegister3 * bool -> ConfigRegister3
val ConfigRegister3_TL_rupd: ConfigRegister3 * bool -> ConfigRegister3
val ConfigRegister3_ULRI_rupd: ConfigRegister3 * bool -> ConfigRegister3
val ConfigRegister3_VEIC_rupd: ConfigRegister3 * bool -> ConfigRegister3
val ConfigRegister3_VInt_rupd: ConfigRegister3 * bool -> ConfigRegister3
val ConfigRegister3_configregister3'rst_rupd:
  ConfigRegister3 * BitsN.nbit -> ConfigRegister3
val ConfigRegister6_LTLB_rupd: ConfigRegister6 * bool -> ConfigRegister6
val ConfigRegister6_TLBSize_rupd:
  ConfigRegister6 * BitsN.nbit -> ConfigRegister6
val ConfigRegister6_configregister6'rst_rupd:
  ConfigRegister6 * BitsN.nbit -> ConfigRegister6
val CauseRegister_BD_rupd: CauseRegister * bool -> CauseRegister
val CauseRegister_ExcCode_rupd:
  CauseRegister * BitsN.nbit -> CauseRegister
val CauseRegister_IP_rupd: CauseRegister * BitsN.nbit -> CauseRegister
val CauseRegister_TI_rupd: CauseRegister * bool -> CauseRegister
val CauseRegister_causeregister'rst_rupd:
  CauseRegister * BitsN.nbit -> CauseRegister
val Context_BadVPN2_rupd: Context * BitsN.nbit -> Context
val Context_PTEBase_rupd: Context * BitsN.nbit -> Context
val Context_context'rst_rupd: Context * BitsN.nbit -> Context
val XContext_BadVPN2_rupd: XContext * BitsN.nbit -> XContext
val XContext_PTEBase_rupd: XContext * BitsN.nbit -> XContext
val XContext_R_rupd: XContext * BitsN.nbit -> XContext
val XContext_xcontext'rst_rupd: XContext * BitsN.nbit -> XContext
val HWREna_CC_rupd: HWREna * bool -> HWREna
val HWREna_CCRes_rupd: HWREna * bool -> HWREna
val HWREna_CPUNum_rupd: HWREna * bool -> HWREna
val HWREna_UL_rupd: HWREna * bool -> HWREna
val HWREna_hwrena'rst_rupd: HWREna * BitsN.nbit -> HWREna
val CP0_BadVAddr_rupd: CP0 * BitsN.nbit -> CP0
val CP0_Cause_rupd: CP0 * CauseRegister -> CP0
val CP0_Compare_rupd: CP0 * BitsN.nbit -> CP0
val CP0_Config_rupd: CP0 * ConfigRegister -> CP0
val CP0_Config1_rupd: CP0 * ConfigRegister1 -> CP0
val CP0_Config2_rupd: CP0 * ConfigRegister2 -> CP0
val CP0_Config3_rupd: CP0 * ConfigRegister3 -> CP0
val CP0_Config6_rupd: CP0 * ConfigRegister6 -> CP0
val CP0_Context_rupd: CP0 * Context -> CP0
val CP0_Count_rupd: CP0 * BitsN.nbit -> CP0
val CP0_Debug_rupd: CP0 * BitsN.nbit -> CP0
val CP0_EPC_rupd: CP0 * BitsN.nbit -> CP0
val CP0_EntryHi_rupd: CP0 * EntryHi -> CP0
val CP0_EntryLo0_rupd: CP0 * EntryLo -> CP0
val CP0_EntryLo1_rupd: CP0 * EntryLo -> CP0
val CP0_ErrCtl_rupd: CP0 * BitsN.nbit -> CP0
val CP0_ErrorEPC_rupd: CP0 * BitsN.nbit -> CP0
val CP0_HWREna_rupd: CP0 * HWREna -> CP0
val CP0_Index_rupd: CP0 * Index -> CP0
val CP0_LLAddr_rupd: CP0 * BitsN.nbit -> CP0
val CP0_PRId_rupd: CP0 * BitsN.nbit -> CP0
val CP0_PageMask_rupd: CP0 * PageMask -> CP0
val CP0_Random_rupd: CP0 * Random -> CP0
val CP0_Status_rupd: CP0 * StatusRegister -> CP0
val CP0_UsrLocal_rupd: CP0 * BitsN.nbit -> CP0
val CP0_Wired_rupd: CP0 * Wired -> CP0
val CP0_XContext_rupd: CP0 * XContext -> CP0
val boolify'32:
  BitsN.nbit ->
  bool *
  (bool *
   (bool *
    (bool *
     (bool *
      (bool *
       (bool *
        (bool *
         (bool *
          (bool *
           (bool *
            (bool *
             (bool *
              (bool *
               (bool *
                (bool *
                 (bool *
                  (bool *
                   (bool *
                    (bool *
                     (bool *
                      (bool *
                       (bool *
                        (bool *
                         (bool *
                          (bool *
                           (bool *
                            (bool * (bool * (bool * (bool * bool))))))))))))))))))))))))))))))
val rec'Index: BitsN.nbit -> Index
val reg'Index: Index -> BitsN.nbit
val write'rec'Index: (BitsN.nbit * Index) -> BitsN.nbit
val write'reg'Index: (Index * BitsN.nbit) -> Index
val rec'Random: BitsN.nbit -> Random
val reg'Random: Random -> BitsN.nbit
val write'rec'Random: (BitsN.nbit * Random) -> BitsN.nbit
val write'reg'Random: (Random * BitsN.nbit) -> Random
val rec'Wired: BitsN.nbit -> Wired
val reg'Wired: Wired -> BitsN.nbit
val write'rec'Wired: (BitsN.nbit * Wired) -> BitsN.nbit
val write'reg'Wired: (Wired * BitsN.nbit) -> Wired
val rec'EntryLo: BitsN.nbit -> EntryLo
val reg'EntryLo: EntryLo -> BitsN.nbit
val write'rec'EntryLo: (BitsN.nbit * EntryLo) -> BitsN.nbit
val write'reg'EntryLo: (EntryLo * BitsN.nbit) -> EntryLo
val rec'PageMask: BitsN.nbit -> PageMask
val reg'PageMask: PageMask -> BitsN.nbit
val write'rec'PageMask: (BitsN.nbit * PageMask) -> BitsN.nbit
val write'reg'PageMask: (PageMask * BitsN.nbit) -> PageMask
val rec'EntryHi: BitsN.nbit -> EntryHi
val reg'EntryHi: EntryHi -> BitsN.nbit
val write'rec'EntryHi: (BitsN.nbit * EntryHi) -> BitsN.nbit
val write'reg'EntryHi: (EntryHi * BitsN.nbit) -> EntryHi
val rec'StatusRegister: BitsN.nbit -> StatusRegister
val reg'StatusRegister: StatusRegister -> BitsN.nbit
val write'rec'StatusRegister: (BitsN.nbit * StatusRegister) -> BitsN.nbit
val write'reg'StatusRegister:
  (StatusRegister * BitsN.nbit) -> StatusRegister
val rec'ConfigRegister: BitsN.nbit -> ConfigRegister
val reg'ConfigRegister: ConfigRegister -> BitsN.nbit
val write'rec'ConfigRegister: (BitsN.nbit * ConfigRegister) -> BitsN.nbit
val write'reg'ConfigRegister:
  (ConfigRegister * BitsN.nbit) -> ConfigRegister
val rec'ConfigRegister1: BitsN.nbit -> ConfigRegister1
val reg'ConfigRegister1: ConfigRegister1 -> BitsN.nbit
val write'rec'ConfigRegister1:
  (BitsN.nbit * ConfigRegister1) -> BitsN.nbit
val write'reg'ConfigRegister1:
  (ConfigRegister1 * BitsN.nbit) -> ConfigRegister1
val rec'ConfigRegister2: BitsN.nbit -> ConfigRegister2
val reg'ConfigRegister2: ConfigRegister2 -> BitsN.nbit
val write'rec'ConfigRegister2:
  (BitsN.nbit * ConfigRegister2) -> BitsN.nbit
val write'reg'ConfigRegister2:
  (ConfigRegister2 * BitsN.nbit) -> ConfigRegister2
val rec'ConfigRegister3: BitsN.nbit -> ConfigRegister3
val reg'ConfigRegister3: ConfigRegister3 -> BitsN.nbit
val write'rec'ConfigRegister3:
  (BitsN.nbit * ConfigRegister3) -> BitsN.nbit
val write'reg'ConfigRegister3:
  (ConfigRegister3 * BitsN.nbit) -> ConfigRegister3
val rec'ConfigRegister6: BitsN.nbit -> ConfigRegister6
val reg'ConfigRegister6: ConfigRegister6 -> BitsN.nbit
val write'rec'ConfigRegister6:
  (BitsN.nbit * ConfigRegister6) -> BitsN.nbit
val write'reg'ConfigRegister6:
  (ConfigRegister6 * BitsN.nbit) -> ConfigRegister6
val rec'CauseRegister: BitsN.nbit -> CauseRegister
val reg'CauseRegister: CauseRegister -> BitsN.nbit
val write'rec'CauseRegister: (BitsN.nbit * CauseRegister) -> BitsN.nbit
val write'reg'CauseRegister: (CauseRegister * BitsN.nbit) -> CauseRegister
val rec'Context: BitsN.nbit -> Context
val reg'Context: Context -> BitsN.nbit
val write'rec'Context: (BitsN.nbit * Context) -> BitsN.nbit
val write'reg'Context: (Context * BitsN.nbit) -> Context
val rec'XContext: BitsN.nbit -> XContext
val reg'XContext: XContext -> BitsN.nbit
val write'rec'XContext: (BitsN.nbit * XContext) -> BitsN.nbit
val write'reg'XContext: (XContext * BitsN.nbit) -> XContext
val rec'HWREna: BitsN.nbit -> HWREna
val reg'HWREna: HWREna -> BitsN.nbit
val write'rec'HWREna: (BitsN.nbit * HWREna) -> BitsN.nbit
val write'reg'HWREna: (HWREna * BitsN.nbit) -> HWREna
val ExceptionCode: ExceptionType -> unit
val SignalException: ExceptionType -> unit
val BYTE: BitsN.nbit
val HALFWORD: BitsN.nbit
val WORD: BitsN.nbit
val DOUBLEWORD: BitsN.nbit
val UserMode: unit -> bool
val SupervisorMode: unit -> bool
val KernelMode: unit -> bool
val BigEndianMem: unit -> bool
val ReverseEndian: unit -> BitsN.nbit
val BigEndianCPU: unit -> BitsN.nbit
val GPR: BitsN.nbit -> BitsN.nbit
val write'GPR: (BitsN.nbit * BitsN.nbit) -> unit
val HI: unit -> BitsN.nbit
val write'HI: BitsN.nbit -> unit
val LO: unit -> BitsN.nbit
val write'LO: BitsN.nbit -> unit
val CPR: (Nat.nat * (BitsN.nbit * BitsN.nbit)) -> BitsN.nbit
val write'CPR:
  (BitsN.nbit * (Nat.nat * (BitsN.nbit * BitsN.nbit))) -> unit
val PSIZE: Nat.nat
val AddressTranslation:
  (BitsN.nbit * (IorD * LorS)) -> (BitsN.nbit * BitsN.nbit)
val LoadMemory:
  (BitsN.nbit * (BitsN.nbit * (BitsN.nbit * (BitsN.nbit * IorD)))) ->
  BitsN.nbit
val StoreMemory:
  (BitsN.nbit *
   (BitsN.nbit * (BitsN.nbit * (BitsN.nbit * (BitsN.nbit * IorD))))) ->
  unit
val Fetch: unit -> (BitsN.nbit option)
val NotWordValue: BitsN.nbit -> bool
val dfn'ADDI: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'ADDIU: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DADDI: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DADDIU: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SLTI: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SLTIU: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'ANDI: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'ORI: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'XORI: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'LUI: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'ADD: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'ADDU: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SUB: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SUBU: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DADD: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DADDU: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DSUB: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DSUBU: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SLT: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SLTU: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'AND: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'OR: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'XOR: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'NOR: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'MOVN: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'MOVZ: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'MADD: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'MADDU: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'MSUB: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'MSUBU: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'MUL: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'MULT: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'MULTU: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'DMULT: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'DMULTU: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'DIV: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'DIVU: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'DDIV: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'DDIVU: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'MFHI: BitsN.nbit -> unit
val dfn'MFLO: BitsN.nbit -> unit
val dfn'MTHI: BitsN.nbit -> unit
val dfn'MTLO: BitsN.nbit -> unit
val dfn'SLL: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SRL: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SRA: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SLLV: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SRLV: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SRAV: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DSLL: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DSRL: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DSRA: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DSLLV: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DSRLV: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DSRAV: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DSLL32: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DSRL32: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DSRA32: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'TGE: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'TGEU: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'TLT: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'TLTU: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'TEQ: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'TNE: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'TGEI: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'TGEIU: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'TLTI: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'TLTIU: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'TEQI: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'TNEI: (BitsN.nbit * BitsN.nbit) -> unit
val loadByte: (BitsN.nbit * (BitsN.nbit * (BitsN.nbit * bool))) -> unit
val loadHalf: (BitsN.nbit * (BitsN.nbit * (BitsN.nbit * bool))) -> unit
val loadWord:
  (bool * (BitsN.nbit * (BitsN.nbit * (BitsN.nbit * bool)))) -> unit
val loadDoubleword:
  (bool * (BitsN.nbit * (BitsN.nbit * BitsN.nbit))) -> unit
val dfn'LB: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'LBU: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'LH: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'LHU: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'LW: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'LWU: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'LL: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'LD: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'LLD: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'LWL: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'LWR: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'LDL: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'LDR: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SB: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SH: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val storeWord: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val storeDoubleword: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SW: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SD: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SC: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SCD: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SWL: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SWR: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SDL: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SDR: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'SYNC: BitsN.nbit -> unit
val dfn'BREAK: unit -> unit
val dfn'SYSCALL: unit -> unit
val dfn'ERET: unit -> unit
val dfn'MTC0: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DMTC0: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'MFC0: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'DMFC0: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'J: BitsN.nbit -> unit
val dfn'JAL: BitsN.nbit -> unit
val dfn'JR: BitsN.nbit -> unit
val dfn'JALR: (BitsN.nbit * BitsN.nbit) -> unit
val ConditionalBranch: (bool * BitsN.nbit) -> unit
val ConditionalBranchLikely: (bool * BitsN.nbit) -> unit
val dfn'BEQ: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'BNE: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'BLEZ: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'BGTZ: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'BLTZ: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'BGEZ: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'BLTZAL: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'BGEZAL: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'BEQL: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'BNEL: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'BLEZL: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'BGTZL: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'BLTZL: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'BGEZL: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'BLTZALL: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'BGEZALL: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'WAIT: unit
val dfn'TLBP: unit -> unit
val dfn'TLBR: unit -> unit
val dfn'TLBWI: unit -> unit
val dfn'TLBWR: unit -> unit
val dfn'CACHE: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> unit
val dfn'RDHWR: (BitsN.nbit * BitsN.nbit) -> unit
val dfn'ReservedInstruction: unit -> unit
val dfn'Unpredictable: unit -> unit
val Run: instruction -> unit
val Decode: BitsN.nbit -> instruction
val Next: unit -> unit
val form1:
  (BitsN.nbit * (BitsN.nbit * (BitsN.nbit * (BitsN.nbit * BitsN.nbit)))) ->
  BitsN.nbit
val form2: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> BitsN.nbit
val form3:
  (BitsN.nbit * (BitsN.nbit * (BitsN.nbit * BitsN.nbit))) -> BitsN.nbit
val form4:
  (BitsN.nbit * (BitsN.nbit * (BitsN.nbit * BitsN.nbit))) -> BitsN.nbit
val form5:
  (BitsN.nbit * (BitsN.nbit * (BitsN.nbit * BitsN.nbit))) -> BitsN.nbit
val form6: (BitsN.nbit * (BitsN.nbit * BitsN.nbit)) -> BitsN.nbit
val Encode: instruction -> BitsN.nbit
val cpr: BitsN.nbit -> string
val r: BitsN.nbit -> string
val c: BitsN.nbit -> string
val i:Nat.nat-> BitsN.nbit -> string
val oi:Nat.nat-> BitsN.nbit -> string
val op1i:Nat.nat-> (string * BitsN.nbit) -> string
val op1r: (string * BitsN.nbit) -> string
val op1ri:Nat.nat-> (string * (BitsN.nbit * BitsN.nbit)) -> string
val op2r: (string * (BitsN.nbit * BitsN.nbit)) -> string
val op2ri:Nat.nat->
  (string * (BitsN.nbit * (BitsN.nbit * BitsN.nbit))) -> string
val op3r: (string * (BitsN.nbit * (BitsN.nbit * BitsN.nbit))) -> string
val op2roi:Nat.nat->
  (string * (BitsN.nbit * (BitsN.nbit * BitsN.nbit))) -> string
val opmem:Nat.nat->
  (string * (BitsN.nbit * (BitsN.nbit * BitsN.nbit))) -> string
val instructionToString: instruction -> string
val skipSpaces: string -> string
val stripSpaces: string -> string
val p_number: string -> (Nat.nat option)
val p_tokens: string -> (string list)
val p_reg: string -> (BitsN.nbit option)
val p_reg2: (string list) -> ((BitsN.nbit * BitsN.nbit) option)
val p_address: string -> ((BitsN.nbit * BitsN.nbit) option)
val p_arg0: string -> maybe_instruction
val p_r1: (string * BitsN.nbit) -> maybe_instruction
val p_r2: (string * (BitsN.nbit * BitsN.nbit)) -> maybe_instruction
val p_r3:
  (string * (BitsN.nbit * (BitsN.nbit * BitsN.nbit))) -> maybe_instruction
val imm_ok:Nat.nat->
  (Nat.nat * (BitsN.nbit * instruction)) -> maybe_instruction
val p_1i: (string * Nat.nat) -> maybe_instruction
val p_r1i: (string * (BitsN.nbit * Nat.nat)) -> maybe_instruction
val p_r2i:
  (string * (BitsN.nbit * (BitsN.nbit * Nat.nat))) -> maybe_instruction
val p_opmem:
  (string * (BitsN.nbit * (BitsN.nbit * BitsN.nbit))) -> maybe_instruction
val instructionFromString: string -> maybe_instruction
val encodeInstruction: string -> string

end