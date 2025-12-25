import React from "react";

const Card = ({ mainTitle, subTitle, explain }) => {
  return (
    <div className="flex flex-col p-2 m-2 rounded-md mx-auto max-w-5xl min-w-sm justify-center bg-neutral-100 shadow">
      <div className="card-body">
        {mainTitle && <h5 className="text-3xl">{mainTitle}</h5>}
        {subTitle && <h6 className="text-2xl">{subTitle}</h6>}
        <p className="text-md whitespace-pre-wrap">{explain}</p>
      </div>
    </div>
  );
};

export default Card;
