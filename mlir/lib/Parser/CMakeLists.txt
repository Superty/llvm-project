set(LLVM_OPTIONAL_SOURCES
  AffineParser.cpp
  AsmParserState.cpp
  AttributeParser.cpp
  DialectSymbolParser.cpp
  Lexer.cpp
  LocationParser.cpp
  Parser.cpp
  Token.cpp
  TypeParser.cpp
  PresburgerParser.cpp
  )

add_mlir_library(MLIRParser
  AffineParser.cpp
  AsmParserState.cpp
  AttributeParser.cpp
  DialectSymbolParser.cpp
  Lexer.cpp
  LocationParser.cpp
  Parser.cpp
  Token.cpp
  TypeParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Parser

  LINK_LIBS PUBLIC
  MLIRIR
  )

add_mlir_library(MLIRPresburgerParser
  PresburgerParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Parser

  LINK_LIBS PUBLIC
  MLIRParser
)
