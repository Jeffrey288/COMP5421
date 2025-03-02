const Mathml2latex = require('mathml-to-latex');

var mathml = `
<mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><mml:msup><mml:mrow><mml:mi>v</mml:mi></mml:mrow><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:msup><mml:mi>H</mml:mi><mml:mi>v</mml:mi><mml:mo>=</mml:mo><mml:mfenced separators="|"><mml:mrow><mml:mtable><mml:mtr><mml:mtd><mml:msub><mml:mrow><mml:mi>v</mml:mi></mml:mrow><mml:mrow><mml:mn>1</mml:mn></mml:mrow></mml:msub></mml:mtd><mml:mtd><mml:msub><mml:mrow><mml:mi>v</mml:mi></mml:mrow><mml:mrow><mml:mn>2</mml:mn></mml:mrow></mml:msub></mml:mtd></mml:mtr></mml:mtable></mml:mrow></mml:mfenced><mml:mfenced separators="|"><mml:mrow><mml:mtable><mml:mtr><mml:mtd><mml:mi>y</mml:mi></mml:mtd></mml:mtr><mml:mtr><mml:mtd><mml:mo>-</mml:mo><mml:mi>x</mml:mi></mml:mtd></mml:mtr></mml:mtable></mml:mrow></mml:mfenced><mml:mfenced separators="|"><mml:mrow><mml:mtable><mml:mtr><mml:mtd><mml:mi>y</mml:mi></mml:mtd><mml:mtd><mml:mo>-</mml:mo><mml:mi>x</mml:mi></mml:mtd></mml:mtr></mml:mtable></mml:mrow></mml:mfenced><mml:mfenced separators="|"><mml:mrow><mml:mtable><mml:mtr><mml:mtd><mml:msub><mml:mrow><mml:mi>v</mml:mi></mml:mrow><mml:mrow><mml:mn>1</mml:mn></mml:mrow></mml:msub></mml:mtd></mml:mtr><mml:mtr><mml:mtd><mml:msub><mml:mrow><mml:mi>v</mml:mi></mml:mrow><mml:mrow><mml:mn>2</mml:mn></mml:mrow></mml:msub></mml:mtd></mml:mtr></mml:mtable></mml:mrow></mml:mfenced><mml:mo>=</mml:mo><mml:msup><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:msub><mml:mrow><mml:mi>v</mml:mi></mml:mrow><mml:mrow><mml:mn>1</mml:mn></mml:mrow></mml:msub><mml:mi>y</mml:mi><mml:mo>-</mml:mo><mml:msub><mml:mrow><mml:mi>v</mml:mi></mml:mrow><mml:mrow><mml:mn>2</mml:mn></mml:mrow></mml:msub><mml:mi>x</mml:mi></mml:mrow></mml:mfenced></mml:mrow><mml:mrow><mml:mn>2</mml:mn></mml:mrow></mml:msup><mml:mo>≤</mml:mo><mml:mn>0</mml:mn></mml:math>
`;
/*
<mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:mi>J</mml:mi></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>W</mml:mi></mml:mrow><mml:mrow><mml:mi>i</mml:mi><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow></mml:mfrac><mml:mo>=</mml:mo><mml:mrow><mml:msubsup><mml:mo stretchy="false">∑</mml:mo><mml:mrow><mml:msup><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mi>'</mml:mi></mml:mrow></mml:msup><mml:mo>=</mml:mo><mml:mn>1</mml:mn></mml:mrow><mml:mrow><mml:msup><mml:mrow><mml:mi>n</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msup></mml:mrow></mml:msubsup><mml:mrow><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:mi>J</mml:mi></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>a</mml:mi></mml:mrow><mml:mrow><mml:msup><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mi>'</mml:mi></mml:mrow></mml:msup></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow></mml:mfrac><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>a</mml:mi></mml:mrow><mml:mrow><mml:msup><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mi>'</mml:mi></mml:mrow></mml:msup></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>W</mml:mi></mml:mrow><mml:mrow><mml:mi>i</mml:mi><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow></mml:mfrac></mml:mrow></mml:mrow><mml:mo>=</mml:mo><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:mi>J</mml:mi></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>a</mml:mi></mml:mrow><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow></mml:mfrac><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>a</mml:mi></mml:mrow><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>W</mml:mi></mml:mrow><mml:mrow><mml:mi>i</mml:mi><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow></mml:mfrac><mml:mo>=</mml:mo><mml:mfenced open="[" close="]" separators="|"><mml:mrow><mml:msubsup><mml:mrow><mml:mi>h</mml:mi></mml:mrow><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup><mml:mo>-</mml:mo><mml:msub><mml:mrow><mml:mi>y</mml:mi></mml:mrow><mml:mrow><mml:mi>j</mml:mi></mml:mrow></mml:msub></mml:mrow></mml:mfenced><mml:msubsup><mml:mrow><mml:mi>h</mml:mi></mml:mrow><mml:mrow><mml:mi>i</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi><mml:mo>-</mml:mo><mml:mn>1</mml:mn></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup><mml:mo>⟹</mml:mo><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:mi>J</mml:mi></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msup><mml:mrow><mml:mi>W</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msup></mml:mrow></mml:mfrac><mml:mo>=</mml:mo><mml:msup><mml:mrow><mml:mi>h</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi><mml:mo>-</mml:mo><mml:mn>1</mml:mn></mml:mrow></mml:mfenced></mml:mrow></mml:msup><mml:msup><mml:mrow><mml:mfenced open="[" close="]" separators="|"><mml:mrow><mml:msup><mml:mrow><mml:mi>h</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msup><mml:mo>-</mml:mo><mml:mi>y</mml:mi></mml:mrow></mml:mfenced></mml:mrow><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:msup></mml:math>
<mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:mi>J</mml:mi></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>b</mml:mi></mml:mrow><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow></mml:mfrac><mml:mo>=</mml:mo><mml:mrow><mml:msubsup><mml:mo stretchy="false">∑</mml:mo><mml:mrow><mml:msup><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mi>'</mml:mi></mml:mrow></mml:msup><mml:mo>=</mml:mo><mml:mn>1</mml:mn></mml:mrow><mml:mrow><mml:msup><mml:mrow><mml:mi>n</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msup></mml:mrow></mml:msubsup><mml:mrow><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:mi>J</mml:mi></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>a</mml:mi></mml:mrow><mml:mrow><mml:msup><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mi>'</mml:mi></mml:mrow></mml:msup></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow></mml:mfrac><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>a</mml:mi></mml:mrow><mml:mrow><mml:msup><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mi>'</mml:mi></mml:mrow></mml:msup></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>b</mml:mi></mml:mrow><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow></mml:mfrac></mml:mrow></mml:mrow><mml:mo>=</mml:mo><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:mi>J</mml:mi></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>a</mml:mi></mml:mrow><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow></mml:mfrac><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>a</mml:mi></mml:mrow><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>b</mml:mi></mml:mrow><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow></mml:mfrac><mml:mo>=</mml:mo><mml:mfenced open="[" close="]" separators="|"><mml:mrow><mml:msubsup><mml:mrow><mml:mi>h</mml:mi></mml:mrow><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup><mml:mo>-</mml:mo><mml:msub><mml:mrow><mml:mi>y</mml:mi></mml:mrow><mml:mrow><mml:mi>j</mml:mi></mml:mrow></mml:msub></mml:mrow></mml:mfenced><mml:mo>⟹</mml:mo><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:mi>J</mml:mi></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msup><mml:mrow><mml:mi>b</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msup></mml:mrow></mml:mfrac><mml:mo>=</mml:mo><mml:msup><mml:mrow><mml:mi>h</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msup><mml:mo>-</mml:mo><mml:mi>y</mml:mi></mml:math>
<mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:mi>J</mml:mi></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>h</mml:mi></mml:mrow><mml:mrow><mml:mi>i</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi><mml:mo>-</mml:mo><mml:mn>1</mml:mn></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow></mml:mfrac><mml:mo>=</mml:mo><mml:mrow><mml:msubsup><mml:mo stretchy="false">∑</mml:mo><mml:mrow><mml:msup><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mi>'</mml:mi></mml:mrow></mml:msup><mml:mo>=</mml:mo><mml:mn>1</mml:mn></mml:mrow><mml:mrow><mml:msup><mml:mrow><mml:mi>n</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msup></mml:mrow></mml:msubsup><mml:mrow><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:mi>J</mml:mi></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>a</mml:mi></mml:mrow><mml:mrow><mml:msup><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mi>'</mml:mi></mml:mrow></mml:msup></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow></mml:mfrac><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>a</mml:mi></mml:mrow><mml:mrow><mml:msup><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mi>'</mml:mi></mml:mrow></mml:msup></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>h</mml:mi></mml:mrow><mml:mrow><mml:mi>i</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi><mml:mo>-</mml:mo><mml:mn>1</mml:mn></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow></mml:mfrac></mml:mrow></mml:mrow><mml:mo>=</mml:mo><mml:mrow><mml:msubsup><mml:mo stretchy="false">∑</mml:mo><mml:mrow><mml:msup><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mi>'</mml:mi></mml:mrow></mml:msup><mml:mo>=</mml:mo><mml:mn>1</mml:mn></mml:mrow><mml:mrow><mml:msup><mml:mrow><mml:mi>n</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msup></mml:mrow></mml:msubsup><mml:mrow><mml:msub><mml:mrow><mml:mfenced open="[" close="]" separators="|"><mml:mrow><mml:msup><mml:mrow><mml:mi>h</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msup><mml:mo>-</mml:mo><mml:mi>y</mml:mi></mml:mrow></mml:mfenced></mml:mrow><mml:mrow><mml:msup><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mi>'</mml:mi></mml:mrow></mml:msup></mml:mrow></mml:msub><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>a</mml:mi></mml:mrow><mml:mrow><mml:msup><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mi>'</mml:mi></mml:mrow></mml:msup></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msubsup><mml:mrow><mml:mi>h</mml:mi></mml:mrow><mml:mrow><mml:mi>i</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi><mml:mo>-</mml:mo><mml:mn>1</mml:mn></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow></mml:mfrac></mml:mrow></mml:mrow><mml:mo>=</mml:mo><mml:mrow><mml:msubsup><mml:mo stretchy="false">∑</mml:mo><mml:mrow><mml:msup><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mi>'</mml:mi></mml:mrow></mml:msup><mml:mo>=</mml:mo><mml:mn>1</mml:mn></mml:mrow><mml:mrow><mml:msup><mml:mrow><mml:mi>n</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msup></mml:mrow></mml:msubsup><mml:mrow><mml:msub><mml:mrow><mml:mfenced open="[" close="]" separators="|"><mml:mrow><mml:msup><mml:mrow><mml:mi>h</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msup><mml:mo>-</mml:mo><mml:mi>y</mml:mi></mml:mrow></mml:mfenced></mml:mrow><mml:mrow><mml:msup><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mi>'</mml:mi></mml:mrow></mml:msup></mml:mrow></mml:msub><mml:msubsup><mml:mrow><mml:mi>W</mml:mi></mml:mrow><mml:mrow><mml:mi>i</mml:mi><mml:msup><mml:mrow><mml:mi>j</mml:mi></mml:mrow><mml:mrow><mml:mi>'</mml:mi></mml:mrow></mml:msup></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msubsup></mml:mrow></mml:mrow><mml:mo>⟹</mml:mo><mml:mfrac><mml:mrow><mml:mo>∂</mml:mo><mml:mi>J</mml:mi></mml:mrow><mml:mrow><mml:mo>∂</mml:mo><mml:msup><mml:mrow><mml:mi>h</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi><mml:mo>-</mml:mo><mml:mn>1</mml:mn></mml:mrow></mml:mfenced></mml:mrow></mml:msup></mml:mrow></mml:mfrac><mml:mo>=</mml:mo><mml:msup><mml:mrow><mml:mi>W</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msup><mml:mfenced open="[" close="]" separators="|"><mml:mrow><mml:msup><mml:mrow><mml:mi>h</mml:mi></mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:msup><mml:mo>-</mml:mo><mml:mi>y</mml:mi></mml:mrow></mml:mfenced></mml:math>


*/

mathml = mathml.replaceAll('mml:', '')
mathml = mathml.replaceAll(` xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"`, '')
mathml = mathml.split('\n')
mathml = mathml.join("\\")
console.log(mathml)

mathml = `<math xmlns:m="m">
      <sSub>
        <sSubPr>
          <ctrlPr>
            </ctrlPr>
        </sSubPr>
        <e>
          <r>
            <t>λ</t>
          </r>
        </e>
        <sub>
          <r>
            <t>n</t>
          </r>
        </sub>
      </sSub>
      <acc>
        <accPr>
          <chr m:val="̃"/>
          <ctrlPr>
            </ctrlPr>
        </accPr>
        <e>
          <sSub>
            <sSubPr>
              <ctrlPr>
                </ctrlPr>
            </sSubPr>
            <e>
              <r>
                <t>x</t>
              </r>
            </e>
            <sub>
              <r>
                <t>n</t>
              </r>
            </sub>
          </sSub>
        </e>
      </acc>
      <r>
        <t>=H</t>
      </r>
      <acc>
        <accPr>
          <chr m:val="̃"/>
          <ctrlPr>
            </ctrlPr>
        </accPr>
        <e>
          <sSub>
            <sSubPr>
              <ctrlPr>
                </ctrlPr>
            </sSubPr>
            <e>
              <r>
                <t>u</t>
              </r>
            </e>
            <sub>
              <r>
                <t>n</t>
              </r>
            </sub>
          </sSub>
        </e>
      </acc>
    </math>`
console.log(Mathml2latex.convert(mathml));
// console.log(mathml)
// for (var i = 0; i < mathml.length; ++i) {
//     console.log(mathml[i]);
// }
