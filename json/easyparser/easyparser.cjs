/*
Utility for Parsing Pdf File in Json Format.
...briefly...instead of using exclusively online tools.

module that parses and converts PDF from binary to json format (https://www.npmjs.com/package/pdf2json)
thanks 

Why: ? : 
JsonBin.io project work well with json data.(Zero Database Setup or Maintenance)
- Website: 	 https://jsonbin.io/features
- API-Reference: https://jsonbin.io/api-reference

*/

import fs from "fs";
import PDFParser from "pdf2json";
const pdfParser = new PDFParser();

pdfParser.on("pdfParser_dataError", errData => console.error(errData.parserError) );

//File amicizia.json HERE
pdfParser.on("pdfParser_dataReady", pdfData => {fs.writeFile("./json/amicizia.json", JSON.stringify(pdfData));});

//File amicizia.pdf HERE 
pdfParser.loadPDF("./pdf/amicizia.pdf");

pdfFilePath="./pdf/";
 fs.readFile(pdfFilePath, (err, pdfBuffer) => {
      if (!err) {
        pdfParser.parseBuffer(pdfBuffer);
      }
    })
